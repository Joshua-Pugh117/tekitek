import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# Lattice definitions (D2Q9)
c = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]])
t = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
cs2 = 1/3  # Speed of sound squared

# Simulation parameters
m = 100
nx, ny = m + 1, m + 1  # Grid size
T_top = 1.0   # Top wall temperature
T_bottom = -1.0  # Bottom wall temperature
D = 0.1  # Thermal diffusivity
omega_s = 1 / (3 * D + 0.5)  # Relaxation parameter for temperature (BGK)
n_steps = 10000  # Number of iterations

# Initialize distribution function for temperature
g = np.zeros((9, nx, ny))
for k in range(9):
    g[k] = 0.0

# Collision step with BGK for temperature
def collision(g):
    chi = np.sum(g, axis=0)  # Macroscopic temperature
    g_eq = t[:, None, None] * chi  # Equilibrium distribution (no advection)
    g_star = g - omega_s * (g - g_eq)  # BGK collision
    return g_star, chi

# Streaming step
def stream(g):
    for k in range(1, 9):  # Skip k=0 (no movement)
        g[k] = np.roll(g[k], shift=c[k], axis=(0, 1))
    return g

# Apply boundary conditions for temperature
def apply_boundary_conditions(g):
    # Periodic boundary conditions on left and right walls
    g[1, 0, :] = g[1, -2, :]
    g[5, 0, :] = g[5, -2, :]
    g[8, 0, :] = g[8, -2, :]
    g[3, -1, :] = g[3, 1, :]
    g[7, -1, :] = g[7, 1, :]
    g[6, -1, :] = g[6, 1, :]
    # Bottom wall (y=0): chi = T_bottom
    chi_bottom = T_bottom
    g[2, :, 0] = 2 * chi_bottom / 9 - g[4, :, 0]
    g[5, :, 0] = chi_bottom / 18 - g[7, :, 0]
    g[6, :, 0] = chi_bottom / 18 - g[8, :, 0]
    # Top wall (y=m): chi = T_top
    chi_top = T_top
    g[4, :, -1] = 2 * chi_top / 9 - g[2, :, -1]
    g[7, :, -1] = chi_top / 18 - g[5, :, -1]
    g[8, :, -1] = chi_top / 18 - g[6, :, -1]
    return g

# Main simulation loop
print("Running heat simulation (BGK)...")
chi_snapshots = []
snapshot_interval = 50
for tn in tqdm(range(n_steps)):
    g, chi = collision(g)
    g = stream(g)
    g = apply_boundary_conditions(g)
    if tn % snapshot_interval == 0:
        chi_snapshots.append(chi.copy())

# Plotting final state
fig, ax = plt.subplots(figsize=(6, 5))
x = np.arange(nx)
y = np.arange(ny)
X, Y = np.meshgrid(x, y)
contour_chi = ax.contourf(X, Y, chi_snapshots[-1].T, levels=20, cmap='hot')
ax.set_title('Final Temperature Distribution')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.colorbar(contour_chi, ax=ax)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, chi_snapshots[-1].T, cmap='hot', edgecolor='none')
ax.set_title('Final Temperature Distribution (3D)')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('Temperature')
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
plt.show()