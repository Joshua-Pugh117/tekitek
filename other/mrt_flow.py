import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.animation as animation

# Lattice definitions (D2Q9)
c = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]])
t = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])

# MRT transformation matrix
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

# Inverse of M
M_inv = np.linalg.inv(M)

# Simulation parameters
m = 100
nx, ny = m + 1, m + 1  # Grid size (101x101 including boundaries)
rho0 = 5.0  # Initial density
v0 = 0.2   # Top wall velocity
nu = 0.02  # Viscosity
omega_m = 1 / (3 * nu + 0.5)  # Relaxation parameter for stress components
s = np.array([0, 1.0, 1.0, 0, 1.0, 0, 1.0, omega_m, omega_m])  # MRT relaxation rates

n_steps = 5000  # Number of iterations

# Initialize distribution function for fluid
f = np.zeros((9, nx, ny))

# Set initial conditions: f to equilibrium with rho=rho0, v=0
for k in range(9):
    f[k] = t[k] * rho0

# Collision step with MRT for fluid
def collision(f):
    # Compute macroscopic variables from f
    rho = np.sum(f, axis=0)
    jx = np.sum(f * c[:, 0, None, None], axis=0)
    jy = np.sum(f * c[:, 1, None, None], axis=0)
    ux = jx / rho
    uy = jy / rho
    
    # MRT collision for f
    # Compute moments
    m = np.einsum('ij,jkl->ikl', M, f)
    
    # Compute equilibrium moments
    u2 = ux**2 + uy**2
    m_eq = np.zeros_like(m)
    m_eq[0] = rho
    m_eq[1] = -2 * rho + 3 * u2
    m_eq[2] = rho - 3 * u2
    m_eq[3] = jx
    m_eq[4] = -jx
    m_eq[5] = jy
    m_eq[6] = -jy
    m_eq[7] = rho * (ux**2 - uy**2)
    m_eq[8] = rho * ux * uy
    
    # Relax moments
    m_star = m - s[:, None, None] * (m - m_eq)
    
    # Post-collision distribution
    f_star = np.einsum('ij,jkl->ikl', M_inv, m_star)
    
    return f_star, rho, ux, uy

# Streaming step
def stream(f):
    for k in range(1, 9):  # Skip k=0 (no movement)
        f[k] = np.roll(f[k], shift=c[k], axis=(0, 1))
    return f

# Apply boundary conditions for fluid
def apply_boundary_conditions(f):
    # Left wall (x=0): Bounce-back
    f[1, 0, :] = f[3, 0, :]
    f[5, 0, :] = f[7, 0, :]
    f[8, 0, :] = f[6, 0, :]
    
    # Right wall (x=m): Bounce-back
    f[3, -1, :] = f[1, -1, :]
    f[7, -1, :] = f[5, -1, :]
    f[6, -1, :] = f[8, -1, :]
    
    # Bottom wall (y=0): Bounce-back
    f[2, :, 0] = f[4, :, 0]
    f[5, :, 0] = f[7, :, 0]
    f[6, :, 0] = f[8, :, 0]
    
    # Top wall (y=m): Zhu and He conditions (moving lid)
    rho_top = f[0, :, -1] + f[1, :, -1] + f[3, :, -1] + 2 * (f[2, :, -1] + f[5, :, -1] + f[6, :, -1])
    f[4, :, -1] = f[2, :, -1]
    f[7, :, -1] = f[5, :, -1] - rho_top * v0 / 6
    f[8, :, -1] = f[6, :, -1] + rho_top * v0 / 6
    
    return f

# Main simulation loop
print("Running flow simulation...")
velocity_snapshots = []  # Store ux and uy instead of psi
snapshot_interval = 100
for tn in tqdm(range(n_steps)):
    # Collision
    f, rho, ux, uy = collision(f)
    f = stream(f)
    f = apply_boundary_conditions(f)
    
    if tn % snapshot_interval == 0:
        velocity_snapshots.append([ux.copy(), uy.copy()])

# Animation setup
fig, ax = plt.subplots(figsize=(6, 5))
x = np.arange(nx)
y = np.arange(ny)
X, Y = np.meshgrid(x, y)

# Initial plot with streamplot
stream = ax.streamplot(X, Y, velocity_snapshots[0][0].T, velocity_snapshots[0][1].T)
ax.set_title('Streamlines')
ax.set_xlabel('x')
ax.set_ylabel('y')

# Update function for animation
def update(frame):
    ax.clear()
    
    # Redraw streamplot with current frame data
    ax.streamplot(X, Y, velocity_snapshots[frame][0].T, velocity_snapshots[frame][1].T)
    
    ax.set_title('Streamlines')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    return ax,

# Create animation
ani = animation.FuncAnimation(fig, update, frames=len(velocity_snapshots), interval=100, blit=False)

plt.show()