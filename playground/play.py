import numpy as np
import matplotlib.pyplot as plt

# --- Parameter ---
L = 10.0          # Länge der Domäne (m)
Nx = 200          # Anzahl Gitterpunkte
dx = L / (Nx - 1)

T = 30.0           # Simulationszeit (s)
dt = 0.005        # Zeitschritt (s)
Nt = int(T / dt)

v = 0.5           # Windgeschwindigkeit (m/s)
D = 0.05          # Diffusionskoeffizient

# --- Gitter ---
grid = np.linspace(0, L, Nx)

# --- Quelle f(x), konstant über die Zeit ---
f = np.zeros(Nx)
source_pos = 3.0    # Position des Emitters (m)
source_width = 0.3  # Breite der Quelle
source_strength = 1.0
f += source_strength * np.exp(-0.5 * ((grid - source_pos) / source_width)**2)

# --- Initialbedingung ---
c = np.zeros(Nx)

# --- Speichern ---
C_time = np.zeros((Nt, Nx))

# --- Diskrete Koeffizienten ---
alpha = v * dt / dx
beta = D * dt / dx**2

# --- Zeitintegration (explizit) ---
for n in range(Nt):
    c_new = c.copy()
    # Advektion (Upwind)
    c_new[1:] -= alpha * (c[1:] - c[:-1])
    # Diffusion (zentral)
    c_new[1:-1] += beta * (c[2:] - 2 * c[1:-1] + c[:-2])
    # Quelle
    c_new += dt * f
    # Randbedingungen (offene Ränder mit perfekter Absaugung)
    c_new[0] = 0.0
    c_new[-1] = 0.0
    # Update
    c = c_new
    C_time[n, :] = c

# --- Sensoren: sparse in Raum und Zeit ---
sensor_positions = [2.0, 5.0, 8.0]
sensor_indices = [np.argmin(np.abs(grid - sp)) for sp in sensor_positions]
time_indices = np.arange(0, Nt, 10)  # jeder 10. Schritt
sensor_data_sparse = C_time[time_indices][:, sensor_indices]

# --- Plot 1: Konzentration über Raum und Zeit ---
plt.figure(figsize=(8,4))
plt.imshow(C_time.T, origin='lower', extent=[0,T,0,L],
           aspect='auto', cmap='viridis')
plt.colorbar(label='Konzentration')
plt.xlabel('Zeit (s)')
plt.ylabel('Position x (m)')
plt.title('Konzentrationsfeld (1D)')
plt.show()

# # --- Plot 2: Sensorzeitreihen ---
# plt.figure(figsize=(8,4))
# for k, sp in enumerate(sensor_positions):
#     plt.plot(np.linspace(0,T,Nt), sensor_data[:,k], label=f"Sensor @ x={sp} m")
# plt.xlabel("Zeit (s)")
# plt.ylabel("Konzentration")
# plt.legend()
# plt.title("Sensor-Zeitreihen")
# plt.show()

# --- Plot 3: Quellprofil ---
plt.figure(figsize=(6,3))
plt.plot(grid, f, label="Quelle f(x)")
plt.xlabel("x (m)")
plt.ylabel("Quellenstärke")
plt.title("Zeitlich konstante Quelle")
plt.legend()
plt.show()

plt.figure()
plt.plot(grid, C_time[-1,:])
plt.xlabel("x (m)")
plt.ylabel("C")
plt.title(f"Finale Verteilung zum Zeitpunkt {T} s ")
plt.show()