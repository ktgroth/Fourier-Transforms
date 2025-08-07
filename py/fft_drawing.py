
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd

df = pd.read_csv('audio.csv')
points = df[['x', 'y']].to_numpy()
complex_points = points[:, 0] + 1j * points[:, 1]

N = len(complex_points)
fourier = np.fft.fft(complex_points) / N
frequencies = np.fft.fftfreq(N, d=1/N)

components = []
for k in range(N):
    freq = frequencies[k]
    coef = fourier[k]
    amplitude = np.abs(coef)
    phase = np.angle(coef)
    components.append((freq, amplitude, phase, coef))

MAX_CYCLES = len(components)
num_cycles = 100

components.sort(key=lambda x: -x[1])

fig, ax = plt.subplots()
fig.patch.set_facecolor('k')
ax.set_facecolor('k')
ax.set_aspect('equal')
ax.axis('off')


title_text = fig.suptitle(f'Vectors: {num_cycles}', color='white', fontsize=12)

full_path = []
pos = 0 + 0j
for frame in range(N):
    t = frame / N
    pos = 0 + 0j
    for freq, amp, phase, _ in components:
        angle = 2 * np.pi * freq * t + phase
        pos += amp * np.exp(1j * angle)
    full_path.append(pos)

background_x = [p.real for p in full_path]
background_y = [p.imag for p in full_path]
ax.plot(background_x, background_y, color='white', alpha=0.3, lw=1)


line, = ax.plot([], [], lw=2, color='cyan')
circles = []
vectors = []

margin = 100
offset_x = -600
offset_y = 0
xs = df['x']
ys = df['y']
ax.set_xlim(xs.min() - margin + offset_x, xs.max() + margin)
ax.set_ylim(ys.min() - margin + offset_y, ys.max() + margin)
connector_line, = ax.plot([], [], color='gray', lw=1, linestyle='--')

path = []
def compute_epicylces(t, cycles):
    pos = 0 + 0j
    positions = [pos]

    for freq, amp, phase, _ in components[:cycles]:
        angle = 2 * np.pi * freq * t + phase
        pos += amp * np.exp(1j * angle)
        positions.append(pos)

    positions = [p + offset_x + offset_y * 1j for p in positions]
    return positions

for _ in range(MAX_CYCLES):
    circle, = ax.plot([], [], 'gray', lw=0.5, alpha=0.6)
    vector, = ax.plot([], [], 'white', lw=1)
    circles.append(circle)
    vectors.append(vector)

def animate(frame):
    t = frame / N
    positions = compute_epicylces(t, num_cycles)

    for i in range(num_cycles):
        center = positions[i]
        tip = positions[i + 1]
        radius = abs(tip - center)

        theta = np.linspace(0, 2 * np.pi, 50)
        circle_x = center.real + radius * np.cos(theta)
        circle_y = center.imag + radius * np.sin(theta)
        circles[i].set_data(circle_x, circle_y)
        vectors[i].set_data([center.real, tip.real], [center.imag, tip.imag])
        circles[i].set_visible(True)
        vectors[i].set_visible(True)

    for i in range(num_cycles, MAX_CYCLES):
        circles[i].set_visible(False)
        vectors[i].set_visible(False)

    connector_line.set_data(
        [positions[-1].real, positions[-1].real - offset_x],
        [positions[-1].imag, positions[-1].imag - offset_y]
    )

    path.append(positions[-1] - offset_x - offset_y * 1j)
    px = [p.real for p in path]
    py = [p.imag for p in path]
    line.set_data(px, py)

    title_text.set_text(f'Vectors: {num_cycles}')
    return [*circles, *vectors, line, title_text, connector_line]

paused = False
def on_click(event):
    global paused

    if paused:
        paused = False
        ani.event_source.start()
    else:
        paused = True
        ani.event_source.stop()

def on_key(event):
    global num_cycles, path

    prev = num_cycles
    if event.key in ['+', '=', 'up']:
        num_cycles = min(num_cycles + 1, MAX_CYCLES)
    elif event.key in ['-', 'down']:
        num_cycles = max(num_cycles - 1, 1)
    else:
        return

    if num_cycles != prev:
        path.clear()
        line.set_data([], [])
        fig.canvas.draw_idle()


fig.canvas.mpl_connect('button_press_event', on_click)
fig.canvas.mpl_connect('key_press_event', on_key)

ani = animation.FuncAnimation(fig, animate, frames=200, interval=20, blit=False, cache_frame_data=False)

try:
    plt.show()
except KeyboardInterrupt:
    pass
finally:
    plt.close(fig)

