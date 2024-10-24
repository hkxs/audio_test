#  Copyright 2024 Hkxs
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the “Software”), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.

import numpy as np
import pyaudio


FRAME_SIZE = 1024
FORMAT = pyaudio.paFloat32
CHANNELS = 1
SAMPLE_RATE = 44100
RECORD_SECONDS = 10

p = pyaudio.PyAudio()
samples = np.arange(FRAME_SIZE) / SAMPLE_RATE

# (44100 / 1024) = 43.07 we can use 430 to get complete cycles
sine_wave = 0.1 * np.sin(2 * np.pi * 430 * samples)

stream = p.open(format=FORMAT, channels=CHANNELS, rate=SAMPLE_RATE, input=True, output=True, frames_per_buffer=FRAME_SIZE)

for i in range(0, int((SAMPLE_RATE // FRAME_SIZE) * RECORD_SECONDS)):
    raw_data = stream.read(FRAME_SIZE)
    data = np.frombuffer(raw_data, dtype=np.float32)
    data = 100 * data + sine_wave
    stream.write(data.T.astype('float32').tobytes())

stream.close()
p.terminate()
