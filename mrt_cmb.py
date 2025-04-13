import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.animation as animation

# Lattice definitions (D2Q9)
c = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]])
t = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])  # Lattice weights
cs2 = 1/3  # Speed of sound squared

# MRT transformation matrix for flow
M = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [-4, -1, -1, -1, -1, 2, 2, 2, 2],
    [4, -2, -2, -2, -2, 1, 1, 1, 1],
    [0, 1, 0, -1, 0, 1, -1, -1, 1],
    [0, -2, 0, 2, 0, 1, -1, -1, 1],
    [0, 0, 1, 0, -1, 1, 1, -1, -1],
    [0, 0, -2, 0, 2, 1, 1, -1, -1],
    [0, 1, -1, 1, -1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, -1, 1, -1]
])
M_inv = np.linalg.inv(M)

# Simulation parameters
m = 100
nx, ny = m + 1, m + 1  # Grid size
rho0 = 1.0  # Reference density
T_hot = 0.5  # Temperature at left wall
T_cold = -0.5  # Temperature at right wall
T_ref = 0.0  # Reference temperature for buoyancy
g_beta = 0.01  # Buoyancy coefficient
nu = 0.1  # Kinematic viscosity
omega_m = 1 / (3 * nu + 0.5)  # Relaxation rate for flow
s = np.array([0, 1.0, 1.0, 0, 1.0, 0, 1.0, omega_m, omega_m])  # MRT relaxation rates
D = 0.1  # Thermal diffusivity
omega_s = 1 / (3 * D + 0.5)  # Relaxation rate for temperature
n_steps = 50_000  # Number of time steps
u_max_threshold = 1  # Stability threshold for velocity



# Initialize distribution functions
f = np.zeros((9, nx, ny))  # Flow distribution
g = np.zeros((9, nx, ny))  # Temperature distribution
for k in range(9):
    f[k] = t[k] * rho0  # Uniform initial density for flow
    g[k] = 0.0  # Zero initial temperature

# Collision step for temperature (BGK with advection)
def collision_g(g, ux, uy):
    chi = np.sum(g, axis=0)  # Macroscopic temperature
    cu = 3 * (c[:, 0, None, None] * ux + c[:, 1, None, None] * uy)  # Velocity term (3 = 1/cs2)
    g_eq = t[:, None, None] * chi * (1 + cu)  # Equilibrium with advection
    g_star = g - omega_s * (g - g_eq)  # BGK collision
    return g_star, chi

# Collision step for flow (MRT with buoyancy force)
def collision_f(f, F_y):
    rho = np.sum(f, axis=0)  # Density
    jx = np.sum(f * c[:, 0, None, None], axis=0)  # x-momentum
    jy = np.sum(f * c[:, 1, None, None], axis=0)  # y-momentum
    ux = jx / rho  # x-velocity
    uy = jy / rho  # y-velocity
    m = np.einsum('ij,jkl->ikl', M, f)  # Moments
    u2 = ux**2 + uy**2
    m_eq = np.zeros_like(m)  # Equilibrium moments
    m_eq[0] = rho
    m_eq[1] = -2 * rho + 3 * u2
    m_eq[2] = rho - 3 * u2
    m_eq[3] = jx
    m_eq[4] = -jx
    m_eq[5] = jy
    m_eq[6] = -jy
    m_eq[7] = rho * (ux**2 - uy**2)
    m_eq[8] = rho * ux * uy
    m_star = m - s[:, None, None] * (m - m_eq)  # Relaxation
    # Add buoyancy force in moment space
    F_k = t[:, None, None] * 3 * c[:, 1, None, None] * F_y[None, :, :]
    m_force = np.einsum('ij,jkl->ikl', M, F_k)
    m_star += m_force
    f_star = np.einsum('ij,jkl->ikl', M_inv, m_star)  # Back to distribution
    return f_star, rho, ux, uy

# Streaming step
def stream(dist):
    for k in range(1, 9):  # Skip k=0 (rest particle)
        dist[k] = np.roll(dist[k], shift=c[k], axis=(0, 1))
    return dist

# Boundary conditions for temperature
def apply_boundary_conditions_g(g):
    # Left wall (x=0): Fixed T_hot
    for k in [1, 5, 8]:
        g[k, 0, :] = t[k] * T_hot + (t[k] - t[8 - k]) * np.sum(g[[0, 2, 4], 0, :], axis=0)
    # Right wall (x=m): Fixed T_cold
    for k in [3, 7, 6]:
        g[k, -1, :] = t[k] * T_cold + (t[k] - t[8 - k]) * np.sum(g[[0, 2, 4], -1, :], axis=0)
    # Bottom wall (y=0): Insulated (zero flux)
    g[2, :, 0] = g[4, :, 0]
    g[5, :, 0] = g[7, :, 0]
    g[6, :, 0] = g[8, :, 0]
    # Top wall (y=m): Insulated (zero flux)
    g[4, :, -1] = g[2, :, -1]
    g[7, :, -1] = g[5, :, -1]
    g[8, :, -1] = g[6, :, -1]
    return g

# Boundary conditions for flow (no-slip walls)
def apply_boundary_conditions_f(f):
    # Left wall (x=0)
    f[1, 0, :] = f[3, 0, :]
    f[5, 0, :] = f[7, 0, :]
    f[8, 0, :] = f[6, 0, :]
    # Right wall (x=m)
    f[3, -1, :] = f[1, -1, :]
    f[7, -1, :] = f[5, -1, :]
    f[6, -1, :] = f[8, -1, :]
    # Bottom wall (y=0)
    f[2, :, 0] = f[4, :, 0]
    f[5, :, 0] = f[7, :, 0]
    f[6, :, 0] = f[8, :, 0]
    # Top wall (y=m)
    f[4, :, -1] = f[2, :, -1]
    f[7, :, -1] = f[5, :, -1]
    f[8, :, -1] = f[6, :, -1]
    return f

# Main simulation loop
print("Running simulation with temperature advection...")
snapshots = []
snapshot_interval = 200
for tn in tqdm(range(n_steps)):
    # Compute macroscopic temperature and buoyancy force
    chi = np.sum(g, axis=0)
    F_y = -g_beta * (chi - T_ref)  # Buoyancy force (vertical)
    # Update flow
    f_star, rho, ux, uy = collision_f(f, F_y)
    # Update temperature with advection
    g_star, _ = collision_g(g, ux, uy)
    # Stream
    f = stream(f_star)
    g = stream(g_star)
    # Apply boundary conditions
    f = apply_boundary_conditions_f(f)
    g = apply_boundary_conditions_g(g)
    # Check stability
    u_max = np.max(np.sqrt(ux**2 + uy**2))
    if u_max > u_max_threshold:
        print(f"Warning: u_max = {u_max:.4f} exceeds threshold at step {tn}")
        break
    # Store snapshots for animation
    if tn % snapshot_interval == 0:
        snapshots.append((chi.copy(), ux.copy(), uy.copy()))

# Animation setup
fig, ax = plt.subplots(figsize=(6, 5))
x = np.arange(nx)
y = np.arange(ny)
X, Y = np.meshgrid(x, y)

# Initial plot
contour = ax.contourf(X, Y, snapshots[0][0].T, levels=20, cmap='turbo_r')
stream = ax.streamplot(X, Y, snapshots[0][1].T, snapshots[0][2].T, color='white')
ax.set_title('Temperature and Flow (Step 0)')
ax.set_xlabel('x')
ax.set_ylabel('y')
# plt.colorbar(contour, ax=ax)

# Animation update function
def update(frame):
    ax.clear()
    contour = ax.contourf(X, Y, snapshots[frame][0].T, levels=20, cmap='turbo_r')
    stream = ax.streamplot(X, Y, snapshots[frame][1].T, snapshots[frame][2].T, color='white')
    ax.set_title(f'Temperature and Flow (Step {frame * snapshot_interval})')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    # return contour.collections + stream.lines,

# Create and display animation
ani = animation.FuncAnimation(fig, update, frames=len(snapshots), interval=50, blit=False)
plt.show()