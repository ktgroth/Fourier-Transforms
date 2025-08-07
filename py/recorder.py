
import pyaudio
import numpy as np
import noisereduce as nr
from scipy.io.wavfile import write

CHANNELS = 1
RATE = 44100
FORMAT = pyaudio.paFloat32
CHUNK = 1024

OUTPUT_FILENAME = 'audio.wav'

p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print('Recording... Press Ctrl+C to stop.')

frames = []

try:
    while True:
        data = stream.read(CHUNK)
        frames.append(np.frombuffer(data, dtype=np.float32))
except KeyboardInterrupt:
    print('Recording stopped.')

stream.stop_stream()
stream.close()
p.terminate()

audio_data = np.hstack(frames)
noise_sample = audio_data[:int(0.5 * RATE)]
reduced_noise = nr.reduce_noise(y=audio_data, y_noise=noise_sample, sr=RATE)
write('audio.wav', RATE, reduced_noise.astype(np.float32))

print(f'Saved recording to {OUTPUT_FILENAME}')
