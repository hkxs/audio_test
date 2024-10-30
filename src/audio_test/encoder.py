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
import time
from pathlib import Path

import librosa
import numpy as np
import pyaudio
import scipy


from pitch_estimation import pitch_estimator


directory = Path(__file__).parent.absolute()
audio_dir = directory.parent.parent / 'audios'
data, sr = librosa.load(audio_dir / "o_sound.wav", mono=False, sr=None)
data = data / np.max(np.abs(data))

pitch = pitch_estimator(data, sr)
print(pitch)  # this should be close to 100Hz
a = librosa.lpc(data, order=10)

period = int(sr // pitch)
impulse_train = scipy.signal.unit_impulse(len(data), range(1, len(data), period))
reconstructed = scipy.signal.lfilter([1], a, impulse_train)
reconstructed = reconstructed / np.max(np.abs(reconstructed))

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paFloat32, channels=1, rate=int(sr), output=True)

stream.write(data.astype('float32').tobytes())
time.sleep(2)
stream.write(reconstructed.astype('float32').tobytes())


stream.close()
p.terminate()

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(data[-512:])
ax.plot(impulse_train[-512:])
ax.plot(reconstructed[-512:])
ax.grid(which='both')
ax.legend(['original', 'impulse train', 'reconstructed'])
plt.show()
