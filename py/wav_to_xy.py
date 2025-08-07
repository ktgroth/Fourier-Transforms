
import numpy as np
from scipy.io import wavfile
import pandas as pd

rate, data = wavfile.read('audio.wav')

if data.ndim > 1:
    data = data[:, 0]

data = data.astype(np.float32)
data /= np.max(np.abs(data))

MAX_POINTS = 300
if len(data) > MAX_POINTS:
    indices = np.linspace(0, len(data) - 1, MAX_POINTS, dtype=int)
    data = data[indices]

x = np.linspace(0, 200, len(data))
y = data * 200

xy_points = np.stack((x, y), axis=-1)

df = pd.DataFrame(xy_points, columns=['x', 'y'])
df.to_csv('audio.csv', index=False)

