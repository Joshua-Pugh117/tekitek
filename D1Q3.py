import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from multiprocessing import Value, Lock

def lb(m):
    N = 100_000
    
    Dt = 1/m
    Dx = Dt
    
    Y = (Dt/Dx)
    K = 0.001
    alpha = 0.25
    w = 1/((K/(alpha*Y*Y*Dt))+0.5)

    f=0.1

    u = np.zeros(m)
    
    f0 = np.zeros(m)
    f1 = np.zeros(m)
    f2 = np.zeros(m)

    for _ in (range(N)):
        
        Tg = f2[0]
        Td = f1[m-1]
        
        for i in range(m-1):
            f1[m-i-1] = f1[m-i-2]
            f2[i] = f2[i+1]
        
        f1[0] = -Tg
        f2[m-1] = -Td
        
        for i in range(m):
            
            f0[i] = f0[i] + Dt * f *(1-alpha)
            f1[i] = f1[i] + Dt*alpha * f/2
            f2[i] = f2[i] + Dt*alpha * f/2
            
            u[i] = f0[i] + f1[i] + f2[i]
            
            f0_eq = u[i] - alpha*u[i]
            f1_eq = u[i]*alpha/2
            f2_eq = u[i]*alpha/2
            
            f0[i] = (1-w)*f0[i] + w*f0_eq
            f1[i] = (1-w)*f1[i] + w*f1_eq
            f2[i] = (1-w)*f2[i] + w*f2_eq
    return u

def lb_moments(m, w2=1.1):
    N = 100_000
    
    Dt = 1/m
    Dx = Dt
    
    Y = (Dt/Dx)
    K = 0.001
    alpha = 0.25
    w1 = 1/((K/(alpha*Y*Y*Dt))+0.5)
    # w2 = 1.1
    # w2 = (8-4*w1)/(4-w1)
    f=0.1

    u = np.zeros(m)
    
    f0 = np.zeros(m)
    f1 = np.zeros(m)
    f2 = np.zeros(m)
    
    q = np.zeros(m)
    e = np.zeros(m)

    for _ in (range(N)):
        
        Tg = f2[0]
        Td = f1[m-1]
        
        for i in range(m-1):
            f1[m-i-1] = f1[m-i-2]
            f2[i] = f2[i+1]
        
        f1[0] = -Tg
        f2[m-1] = -Td
        
        for i in range(m):
            
            # f0[i] = f0[i] + Dt * f *(1-alpha)
            # f1[i] = f1[i] + Dt*alpha * f/2
            # f2[i] = f2[i] + Dt*alpha * f/2
            
            
            # moments
            u[i] = f0[i] + f1[i] + f2[i]
            u[i] = u[i] + Dt*f
            q[i] = (f1[i] - f2[i])*Y
            e[i] = (f1[i] + f2[i])*Y*Y/2
            
            q[i] = (1-w1)*q[i] 
            e[i] = (1-w2)*e[i] + w2*(alpha*u[i]*Y*Y)/2
            
            f0[i] = u[i] - 2 * e[i]/(Y*Y)
            f1[i] = q[i]/2/Y + e[i]/Y/Y
            f2[i] = -q[i]/2/Y + e[i]/Y/Y
    return u
    


def exact(m):
    
    f = 0.1
    Dt = 1/m
    Dx = Dt
    
    K = 0.001
    
    x = np.linspace(Dx/2,1-Dx/2,m)
    return f*x*(1-x)/2/K
    



def error(i, w2=1.1):
    # print(f"Calculating error for m = {i}")
    u = lb_moments(i, w2)
    u_exact = exact(i)
    # return (w2, (1/i)*np.linalg.norm(np.abs(u - u_exact)))
    # return (w2, np.mean(np.abs(u - u_exact)))
    return (i, np.min(np.abs(u - u_exact)))

if __name__ == '__main__':
    # with Pool() as pool:
    #     errors = pool.map(error, range(3, 80,5))

    # errors = np.array(errors)
    # np.save('errors1.npy', errors)
    
    # u_moments = lb_moments(20)
    # # u = lb(30)
    # u_exact = exact(20)
    # x = np.linspace(0,1,20)
    # # print(error(30))
    # plt.figure()
    # # # plt.plot(x, u, label='LB')
    # plt.plot(x, u_exact, label='Exact')
    # plt.plot(x, u_moments, label='Moments')
    # plt.legend()
    # plt.show()
    
    

    # plt.figure()
    # plt.plot(np.log(errors[:,0]), np.log(errors[:,1]))
    # coeff = np.polyfit(np.log(errors[:,0]), np.log(errors[:,1]), 1)
    # print(f"Polynomial coefficients: {coeff}")
    
    # plt.savefig('erreur_diffusion.png')

    # plt.show()

    
    with Pool() as pool:
        errors = pool.map(error, range(3, 50, 5))

    errors = np.array(errors)
    # np.save('errors2.npy', errors)

    # errors = np.load('other/errors.npy')
    
    plt.figure(figsize=(10, 5))
    plt.plot(np.log(errors[:, 0]), np.log(errors[:, 1]))
    coeff = np.polyfit(np.log(errors[:, 0]), np.log(errors[:, 1]), 1)
    print(f"Polynomial coefficients: {coeff}")

    plt.xlabel('log(m)')
    plt.ylabel('log(Error)')
    
    # Set custom x-tick labels
    plt.xticks(ticks=np.log(errors[:, 0]), labels=errors[:, 0].astype(int))
    
    plt.savefig('erreur_diffusion.png')
    plt.show()
    
    # w2_values = np.linspace(0.1, 2.0, 50)
    # errors_list = []

    # with Pool() as pool:
    #     errors_list = pool.starmap(error, [(40, w2) for w2 in w2_values])
        
    # errors_list = np.array(errors_list)

    # plt.figure(figsize=(10, 5))
    # plt.plot((errors_list[:, 0]), (errors_list[:, 1]), label='Erreur vs S2')
    # # plt.plot(np.log(errors_list[:, 0]), np.log(errors_list[:, 1]), label='Error vs m')
    # # plt.xticks(ticks=np.log(errors_list[:, 0]), labels=np.round(errors_list[:, 0].astype(float), 2), rotation=45)
    
    # min_error_index = np.argmin(errors_list[:, 1])
    # min_w2 = errors_list[min_error_index, 0]
    # plt.axvline(x=min_w2, color='r', linestyle='--', label=f'Min erreur S2={min_w2:.2f}')

    # plt.xlabel('S2')
    # plt.ylabel('Erreur')
    # plt.title('Erreur en fonction de S2')
    # plt.legend()
    # plt.savefig('diffusion_S2.png')
    # plt.show()