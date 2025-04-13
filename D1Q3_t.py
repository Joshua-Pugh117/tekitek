import numpy as np
import matplotlib.pyplot as plt

# Clear all variables and close figures not needed in Python
# clc and clf are not necessary in Python

L = 1000
Nx = 50
al = 1
a = 1000
ro0 = 1000
v0 = 1

# Calculations
dx = L / Nx
XX = np.arange(dx/2, L, dx)  # Similar to MATLAB's [dx/2:dx:L-dx/2]
dt = dx * np.sqrt(al) / a
lambda_ = dx / dt  # Using lambda_ instead of lambda to avoid Python keyword
nu = 1e-6
# nu = 1.99
s = 1 / ((nu / ((1-al) * lambda_ * dx)) + 0.5)
s = 1.1

# Matrix definitions
MM = np.array([[1, 1, 1],
               [0, lambda_, -lambda_],
               [0, lambda_**2/2, lambda_**2/2]])
Minv = np.linalg.inv(MM)

P0 = ro0 * 9.81 * (Nx-1)  # 150
rold = ro0 * np.ones(Nx)  # -0.02*XX not included as in MATLAB
qold = ro0 * np.ones(Nx)
eeeq = al * lambda_**2 * rold / 2

# Initialize ffold
ffold = np.zeros((3, Nx))
for i in range(Nx):
    ffold[:, i] = Minv @ np.array([rold[i], qold[i], eeeq[i]])

T = 6 * L / a
Nt = int(np.round(T/dt))

# Time loop
SolP = np.zeros(Nt)
rhonew = np.zeros(Nx)
qnew = np.zeros(Nx)
ffint = np.zeros((3, Nx))
ffnew = np.zeros((3, Nx))

for k in range(Nt):
    for i in range(Nx):  # Space loop
        mold = MM @ ffold[:, i]
        rhonew[i] = mold[0]
        qnew[i] = mold[1]
        
        mms = np.zeros(3)
        mms[0] = mold[0]
        mms[1] = mold[1]
        mms[2] = mold[2] + s * ((lambda_**2/2) * al * mold[0] + 
                              (mold[1]*mold[1])/(2*mold[0]) - mold[2])
        
        ffint[:, i] = Minv @ mms

    SolP[k] = (rhonew[-1] - ro0) * a * a
    
    jout = 0
    for i in range(Nx):
        # Advection + Boundary Conditions
        if i == 0:
            ffnew[:, i] = [ffint[0, i],
                          -ffint[2, 0] + al * ro0,
                          ffint[2, 1]]
        elif i == Nx-1:
            ffnew[:, i] = [ffint[0, i],
                          ffint[1, i-1],
                          ffint[1, Nx-1] - jout]
        else:
            ffnew[:, i] = [ffint[0, i],
                          ffint[1, i-1],
                          ffint[2, i+1]]
    
    ffold = ffnew.copy()

# Plotting
plt.figure(5)
plt.plot(SolP[:-1]/ro0/a/v0, 'r-')
plt.title(f"al = {al}, s = {s}")
plt.savefig(f'pressure_{al}_{s}.png')
plt.show()

# Commented plotting sections from MATLAB:
# plt.subplot(1, 2, 1)
# plt.plot(XX, qnew)
# plt.title('VV')
# plt.subplot(1, 2, 2)
# plt.plot(XX, rhonew)
# plt.title('PP')