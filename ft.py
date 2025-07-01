
import matplotlib.pyplot as plt
import numpy as np


def fft(signal):
    N = len(signal)

    if N <= 1:
        return
    
    even = [0 for i in range(N//2)]
    odd = [0 for i in range(N//2)]

    for i in range(N//2):
        even[i] = signal[2 * i]
        odd[i] = signal[2 * i + 1]

    fft(even)
    fft(odd)

    for k in range(N//2):
        s = odd[k] * np.exp(-2j * np.pi * k / N)
        signal[k] = even[k] + s
        signal[k + N//2] = even[k] - s


def next_power_of_2(N):
    i = 1
    while i < N:
        i *= 2

    return i


Fs = 1000
T = 1 / Fs
L = 1024  # keep it small — DFT is O(N²)
t = np.linspace(0, (L-1)*T, L)
signal = 0.7 * np.sin(2*np.pi*50*t) + np.sin(2*np.pi*120*t)

signal = list(map(complex, signal))

print(t)

# Our DFT
fft(signal)
freqs = np.fft.fftfreq(L, T)  # to match frequency axis

# Positive frequencies
plt.plot(freqs[:L//2], 2.0/L * np.abs(signal[:L//2]))
plt.title("Magnitude Spectrum (Custom DFT)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.grid()
plt.show()
