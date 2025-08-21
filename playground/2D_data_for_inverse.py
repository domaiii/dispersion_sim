import numpy as np
import matplotlib.pyplot as plt

# --- Parameter ---
Lx = 10.0
Ly = 10.0
Nx = 200
Ny = 200
dx = Lx / (Nx - 1)
dy = Ly / (Ny - 1)

T = 30.0
dt = 0.005
Nt = int(T / dt)

vx = 0.5
vy = 0.2
D = 0.05

# --- Gitter ---
x_grid = np.linspace(0, Lx, Nx)
y_grid = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x_grid, y_grid)

# --- Quelle f(x,y), konstant über die Zeit ---
f = np.zeros((Ny, Nx))
source_pos_x = 3.0
source_pos_y = 5.0
source_width_x = 0.3
source_width_y = 0.3
source_strength = 10.0
f += source_strength * np.exp(-0.5 * (((X - source_pos_x) / source_width_x)**2 + ((Y - source_pos_y) / source_width_y)**2))

# --- Initialbedingung ---
c = np.zeros((Ny, Nx))

# --- Speichern für Plots ---
plot_times = np.linspace(0, T, 6, endpoint=True)
plot_indices = [int(t / dt) for t in plot_times]
plot_indices[-1] = Nt - 1 # sicherstellen, dass der letzte Zeitpunkt erfasst wird
plot_data = []

# --- Diskrete Koeffizienten ---
alpha_x = vx * dt / dx
alpha_y = vy * dt / dy
beta_x = D * dt / dx**2
beta_y = D * dt / dy**2

# --- Zeitintegration (explizit) ---
for n in range(Nt):
    c_new = c.copy()

    # Advektion (Upwind-Schema)
    if vx > 0:
        c_new[:, 1:] -= alpha_x * (c[:, 1:] - c[:, :-1])
    else: # falls Wind von rechts kommt
        c_new[:, :-1] -= alpha_x * (c[:, :-1] - c[:, 1:])

    if vy > 0:
        c_new[1:, :] -= alpha_y * (c[1:, :] - c[:-1, :])
    else: # falls Wind von oben kommt
        c_new[:-1, :] -= alpha_y * (c[:-1, :] - c[1:, :])

    # Diffusion (zentral)
    c_new[1:-1, 1:-1] += beta_x * (c[1:-1, 2:] - 2 * c[1:-1, 1:-1] + c[1:-1, :-2])
    c_new[1:-1, 1:-1] += beta_y * (c[2:, 1:-1] - 2 * c[1:-1, 1:-1] + c[:-2, 1:-1])

    # Quelle
    c_new += dt * f

    # Randbedingungen (offene Ränder, perfekte Absaugung)
    c_new[:, 0] = 0.0 # linker Rand
    c_new[:, -1] = 0.0 # rechter Rand
    c_new[0, :] = 0.0 # unterer Rand
    c_new[-1, :] = 0.0 # oberer Rand

    # Update
    c = c_new

    # Daten für Plot speichern
    if n in plot_indices:
        plot_data.append(c.copy())


# --- Visualisierung ---
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, (ax, data) in enumerate(zip(axes, plot_data)):
    im = ax.imshow(data, origin='lower', extent=[0, Lx, 0, Ly], cmap='viridis', vmin=0, vmax=np.max(c))
    ax.set_title(f"Zeit = {plot_times[i]:.2f} s")
    ax.set_xlabel("Position x (m)")
    ax.set_ylabel("Position y (m)")
    
    # Quelle hervorheben
    ax.plot(source_pos_x, source_pos_y, 'ro', markersize=8, label='Quelle')
    if i == 0:
        ax.legend()
        
fig.suptitle('Konzentrationsverteilung in 2D über die Zeit', fontsize=16)
fig.colorbar(im, ax=axes, orientation='vertical', fraction=.05, pad=.04, label='Konzentration')
plt.show()