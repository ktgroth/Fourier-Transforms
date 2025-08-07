
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pyaudio

CHUNK = 1024
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 44100


pa = pyaudio.PyAudio()
#for i in range(pa.get_device_count()):
#    info = pa.get_device_info_by_index(i)
#    print(info)
#    print(f"{i}: {info['name']} (Input Channels: {info['maxInputChannels']}, Default Sample Rate: {info['defaultSampleRate']})")


stream = pa.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

mode = input("Select mode: 'bar' for bar graph, 'spec' for spectrogram: ").strip().lower()

freqs = np.fft.rfftfreq(CHUNK, d=1.0/RATE)
rolling_max = 0.01
threshold = 1e-3

if mode == "bar":
    width = freqs[1]-freqs[0]

    fig, ax = plt.subplots()
    bars = ax.bar(freqs, np.zeros_like(freqs), width=width)
    ax.set_xlim(max(freqs[0], 20), freqs[-1])
    ax.set_ylim(0, 0.01)
    ax.set_xscale('log')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Real-Time FFT (Bar Graph)')

    annotations = [ax.text(0, 0, '', fontsize=8, color='red') for _ in range(5)]

    def update_bar(frame):
        data = stream.read(CHUNK, exception_on_overflow=False)
        samples = np.frombuffer(data, dtype=np.float32)

        fft_data = np.abs(np.fft.rfft(samples))
        fft_data[fft_data < threshold] = threshold
        fft_data = np.log10(fft_data)

        top_indicies = np.argpartition(fft_data, -5)[-5:]
        top_indicies = top_indicies[np.argsort(-fft_data[top_indicies])]

        for i, txt in enumerate(annotations):
            freq = freqs[top_indicies[i]]
            amp = fft_data[top_indicies[i]]
            txt.set_position((freq, amp))
            txt.set_text(f"{int(freq)} Hz\n{amp:.4f}")
            txt.set_visible(True)

        for bar, height in zip(bars, fft_data):
            bar.set_height(height)

        global rolling_max

        peak = np.max(fft_data)
        rolling_max = 0.8 * rolling_max + 0.2 * peak
        ax.set_ylim(0, max(0.01, rolling_max))

        return [bars] + annotations

    ani = animation.FuncAnimation(fig, update_bar, interval=30, blit=False, cache_frame_data=False)

elif mode == "spec":
    history_len = 256
    spec_data = np.zeros((history_len, len(freqs)))

    fig, ax = plt.subplots()
    im = ax.imshow(
        spec_data,
        origin='lower',
        aspect='auto',
        extent=[freqs[0], freqs[-1], 0, history_len],
        interpolation='auto',
        cmap='turbo',
        animated=True
    )
    ax.set_yscale('linear')
    ax.set_xlim(max(freqs[0], 20), freqs[-1])
    ax.set_xscale('log')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Time')
    ax.invert_yaxis()
    ax.set_title('Real-Time Spectrogram')

    def update_spec(frame):
        data = stream.read(CHUNK, exception_on_overflow=False)
        samples = np.frombuffer(data, dtype=np.float32)

        fft_data = np.abs(np.fft.rfft(samples))
        fft_data = np.maximum(fft_data, threshold)
        fft_data = np.log10(fft_data)

        spec_data[:-1] = spec_data[1:]
        spec_data[-1, :] = fft_data

        im.set_data(spec_data)
        im.set_clim(np.log10(threshold), 0)
        return [im]

    ani = animation.FuncAnimation(fig, update_spec, interval=30, blit=True, cache_frame_data=False)

else:
    fig, ax = plt.subplots()
    x = np.fft.rfftfreq(CHUNK, 1.0 / RATE)
    line, = ax.plot(x, np.zeros_like(x))
    ax.set_xlim(max(freqs[0], 20), freqs[-1])
    ax.set_ylim(0, 0.01)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Real-Time Audio FFT")

    annotations = [ax.text(0, 0, '', fontsize=8, color='red') for _ in range(5)]

    def update(frame):
        data = stream.read(CHUNK, exception_on_overflow=False)
        samples = np.frombuffer(data, dtype=np.float32)

        fft_data = np.abs(np.fft.rfft(samples))
        fft_data[fft_data < threshold] = threshold

        top_indicies = np.argpartition(fft_data, -5)[-5:]
        top_indicies = top_indicies[np.argsort(-fft_data[top_indicies])]

        for i, txt in enumerate(annotations):
            freq = freqs[top_indicies[i]]
            amp = fft_data[top_indicies[i]]
            txt.set_position((freq, amp))
            txt.set_text(f"{int(freq)} Hz\n{amp:.4f}")
            txt.set_visible(True)

        line.set_ydata(fft_data)
        global rolling_max

        peak = np.max(fft_data)
        rolling_max = 0.8 * rolling_max + 0.2 * peak
        ax.set_ylim(0, max(0.01, rolling_max))

        return [line] + annotations

    ani = animation.FuncAnimation(fig, update, interval=20, blit=True, cache_frame_data=False)

print("Recording... Press Ctrl+C to stop.")
try:
    plt.show()
except KeyboardInterrupt:
    pass
finally:
    stream.stop_stream()
    stream.close()
    pa.terminate()

