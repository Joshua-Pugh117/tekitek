import matplotlib.pyplot as plt
import numpy as np

def lb():
    w = 4/3

    m = 100
    N = 200

    u = np.zeros(m+1)
    u1 = u/2
    u2 = u/2

    for _ in range(N):
        
        
        u1_eq = u/2
        u2_eq = u/2
        
        u1_e = u1*(1-w) + w*u1_eq
        u2_e = u2*(1-w) + w*u2_eq
        
        u1[1:] = u1_e[:-1]
        u2[:-1] = u2_e[1:]
        
        for i in range(1, m):
            u1[m-i] = u1_e[m-i-1]
            u2[i-1] = u2_e[i]
        
        # u1[m-1] = u1[m-2]
        # u2[m-1] = u2[m-2]
        
        u1[0] = 1 - u2[0]
        u = u1 + u2
        
    return u

def df():
    o = 1
    T = 1/2
    D = 1/4

    m = 100
    N = 400
    
    U0 = 1
    
    u = np.zeros(m+1)

    for _ in range(N+1):
        # u = 
        u[-1] = u[-2]
        u[0] = U0

    return u



plt.plot(lb())
# plt.plot(df(), linestyle='dashed' )
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Plot of u')
plt.ylim(0, 1)
# plt.xlim(0, m)
plt.show()

    