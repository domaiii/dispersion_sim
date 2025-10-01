import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.interpolate import make_interp_spline
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib

n = 250
x = np.linspace(-1.5, 1.5, n)
y = np.linspace(0, 3, n)
X, Y = np.meshgrid(x, y)

# 1. Smooth plume
np.random.seed(10)
width = 0.15 + 0.3 * (Y / Y.max())
random_knots = np.random.randn(12) * 0.1
y_knots = np.linspace(0, 3, len(random_knots))
spline = make_interp_spline(y_knots, random_knots, k=2)
random_path = spline(Y[:, 0])
centerline = np.tile(random_path, (n, 1)).T
smooth = np.exp(-((X - centerline) ** 2) / (2 * width**2)) * np.exp(-Y / 2)
smooth = gaussian_filter(smooth, sigma=1.2)

# 2. Turbulent jet
np.random.seed(25)
turbulent = np.zeros_like(X)
cx_prev = 0.0
mask_prev = np.ones(n)
hole_prev = np.zeros(n)

for j in range(n):
    y_pos = Y[j,0]
    cx = cx_prev + 0.03 * np.random.randn() #+ (0.1 + y_pos*0.01) * np.sin(0.02)
    cx_prev = cx
    width_j = 0.05 + 0.5 * (j / n)
    base = 3.5 * np.exp(-((X[j, :] - cx) ** 2) / (2 * width_j**2))
    
    mask = (np.random.rand(n) > (0.7 - 0.4*j/n)).astype(float)
    mask = np.clip(0.5*mask_prev + 0.5*mask, 0, 1)
    mask_prev = mask
    
    row = base * mask
    
    if y_pos > 0.8:
        for _ in range(np.random.randint(1,3)):
            hotspot_x = np.random.uniform(-0.5,0.5)
            hotspot_sigma = 0.05 + 0.05*np.random.rand()
            hotspot_amplitude = 0.5 + 0.5*np.random.rand()
            row += hotspot_amplitude * np.exp(-(X[j,:] - hotspot_x)**2 / (2*hotspot_sigma**2))

        # Anzahl der Zentren steigt mit y
        max_centers = 1 + int(3*(y_pos-0.8)/(3-0.8))  # steigt von 1 bis 4 oben
        n_centers = np.random.randint(1, max_centers + 1)
        row_temp = np.zeros_like(row)
        for _ in range(n_centers):
            cx_center = np.random.uniform(-0.5, 0.5)
            sigma_center = 0.05 + 0.05*np.random.rand()
            amplitude_center = 0.8 + 0.5*np.random.rand()
            row_temp += amplitude_center * np.exp(-(X[j,:]-cx_center)**2/(2*sigma_center**2))
        row += row_temp
    if y_pos < 0.0:
        hole_mask = (np.random.rand(n) < 1.5)
        row[hole_mask] *= np.random.uniform(0.0,0.5, size=hole_mask.sum())

    y_pos = Y[j,0]
    # Basisplume + Zentren wie gehabt
    row = base * mask * np.exp(-y_pos / 1.5) + row

    # Kohärente große Löcher oben
    if y_pos > 0.2:
        hole_mask = np.zeros(n)
        # vererbe vorherige Löcher
        hole_mask = 0.9 * hole_prev
        # füge wenige neue große Löcher hinzu
        for _ in range(np.random.randint(1, 3)):  # 1–2 Löcher pro Zeile
            center = np.random.randint(0, n)
            width = np.random.randint(15, 30)      # Breite des Lochs in Pixeln
            start = max(center - width//2, 0)
            end = min(center + width//2, n)
            hole_mask[start:end] = 1.0
        row *= (1 - hole_mask)  # wende Löcher an
        hole_prev = hole_mask  # für die nächste Zeile speichern

    turbulent[j, :] = row * np.exp(-y_pos / 1.5)

turbulent = gaussian_filter(turbulent, sigma=2)

# Diskrete Farbskala
colors = ["#0556A6", "#1B82EA", "#269DD8", "#FFFA99", "#FFCC66", "#CC3300"]
cmap = ListedColormap(colors)
levels = np.linspace(0, 1, len(colors)+1)
norm = BoundaryNorm(levels, ncolors=cmap.N)

# Plotten 
from scipy.ndimage import zoom

zoom_factor = 3
smooth_hi = zoom(smooth, zoom=zoom_factor, order=3)
turbulent_hi = zoom(turbulent, zoom=zoom_factor, order=3)

fig1, ax1 = plt.subplots(figsize=(6,6), dpi=300)
im1 = ax1.imshow(smooth_hi, origin='lower', cmap=cmap, norm=norm,
                 extent=[-1.5,1.5,0,3], aspect='equal')
ax1.axis('off')
#fig1.text(0.5, 0.92, "Uniform dispersion", fontsize=14, ha='center')
#fig1.colorbar(im1, ax=ax1, shrink=0.8, label="Relative concentration")
fig1.savefig("smooth_plume.svg", format='svg', bbox_inches='tight')

fig2, ax2 = plt.subplots(figsize=(6,6), dpi=300)
im2 = ax2.imshow(turbulent_hi, origin='lower', cmap=cmap, norm=norm,
                 extent=[-1.5,1.5,0,3], aspect='equal')
ax2.axis('off')
#fig2.text(0.5, 0.92, "Irregular dispersion", fontsize=14, ha='center')
fig2.colorbar(im2, ax=ax2, shrink=0.6, label="Rel. concentration")
fig2.savefig("turbulent_plume.svg", format='svg', bbox_inches='tight')




