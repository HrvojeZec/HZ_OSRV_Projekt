import time
import random
from tkinter import TclError
import pyaudio
import wave
import sys
import struct
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from scipy.fftpack import fft


# number of audio samples per frame displayed
# 4096 samples per chunk
CHUNK = 1024*4  # lower chunk size = more samples per frame, increases refresh rate
FORMAT = pyaudio.paInt16  # bytes per sample (audio format)
CHANNELS = 1  # monosound
RATE = 44100  # sampling frequency in hz, number of samples per second
# open the file for reading.
audio_path = "file_example_WAV_1MG.wav"
wf = wave.open(audio_path, 'rb')

# create an audio object
p = pyaudio.PyAudio()


stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(), rate=wf.getframerate(), input=True,
                output=True, frames_per_buffer=1024)
files_seconds = wf.getnframes()/RATE  # length of wav file


def y_range(filename):
    wf = wave.open(filename, 'rb')
    x = []
    count = 0
    while count <= (int(wf.getnframes()/CHUNK)):
        data = wf.readframes(CHUNK)  # read 1 chunk
        data_int = np.frombuffer(data, dtype=np.int16)
        x.append(data_int)
        count = count+1

    result = []
    for list in x:
        result.append(min(list))
        result.append(max(list))

    y_max = np.amax(result)
    y_min = np.amin(result)
    return y_max, y_min


mpl.style.use('seaborn')
sns.set(style="darkgrid")
fig, (ax, ax2) = plt.subplots(2)

x = np.arange(0, 16*CHUNK, 8)  # step size 2
x_fft = np.linspace(0, RATE, CHUNK)

line, = ax.plot(x, (np.random.rand(2*CHUNK)), '-', lw=1)
# change to log scale
line_fft, = ax2.semilogx(x_fft, np.random.rand(CHUNK), '-', lw=1)

ax.set_ylim(y_range(audio_path))
ax.set_xlim(0, CHUNK/2)
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax.yaxis.set_ticklabels([])
ax.patch.set_facecolor('black')
ax.patch.set_alpha(.5)


ax2.set_xlim(20, RATE/2)
ax2.get_xaxis().set_visible(False)
ax2.yaxis.set_ticklabels([])
ax2.patch.set_facecolor('black')
ax2.patch.set_alpha(.5)  # transparency

line.set_color("white")
line_fft.set_color("white")

colors = ["#330099", "#333399", "#3300CC", "#3333CC", "#333099",
          "#333999", "#400099", "#260099"]


frame_count = 0
start_time = time.time()
fig.show()

while True:
    data = wf.readframes(CHUNK)  # read 1 chunk

    data_int = np.fromstring(data, dtype=np.int16)

    line.set_ydata(data_int)

    y_fft = fft(data_int)
    # slice and rescale
    line_fft.set_ydata(np.abs(y_fft[0:CHUNK])*2 / (10000 * CHUNK))

    try:
        fig.canvas.draw()
        fig.canvas.flush_events()

        # write to stream to play sound
        stream.write(data)
        frame_count = frame_count + 1

        # random background color
        ax.patch.set_facecolor(random.choice(colors))
        ax2.patch.set_facecolor(random.choice(colors))

    except TclError:
        frame_rate = frame_count/(time.time() - start_time)
        print("stream stopped")
        print('average frame rate = {:.0f} FPS'.format(frame_rate))
        break


plt.plot((fft(np.sin(2 * x))))
