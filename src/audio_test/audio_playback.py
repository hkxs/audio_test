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

import librosa
import pyaudio
import datetime

files = [
    librosa.example("nutcracker"),
    librosa.example("trumpet", hq=True),
    librosa.example("brahms", hq=True),
    librosa.example("humpback", hq=True),
]
p = pyaudio.PyAudio()

for filename in files:
    print("*" * 80)
    print(filename)

    sr = librosa.get_samplerate(filename)
    print(f"Sample Rate: {sr}")

    frames = librosa.stream(filename,
                            block_length=1,
                            frame_length=int(1024 * sr),
                            hop_length=int(1024 * sr),
                            mono=False
                            )

    first_frame = next(frames)  # get the channels information from the first frame
    num_channels, num_samples = first_frame.shape if first_frame.ndim > 1 else (1, first_frame.shape[0])
    duration = datetime.timedelta(seconds=(num_samples / sr))

    print(f"Channels: {num_channels}")
    print(f"Duration: {duration}")

    stream = p.open(format=pyaudio.paFloat32, channels=num_channels, rate=sr, output=True)

    print("Playing stream")
    # we need to play the first frame and then the rest of the frames
    stream.write(first_frame.T.astype('float32').tobytes())
    for frame in frames:
        stream.write(frame.T.astype('float32').tobytes())
    print("Closing stream")
    stream.close()

p.terminate()
