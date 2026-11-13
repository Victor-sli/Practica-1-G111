import numpy as np
import matplotlib.pyplot as plt
import math

# Definim els paràmetres del problema

alpha = 0.56/(3698*1081)

Nx = 101

# Definim els delta x i delta t
dx = 1/(Nx-1)
dt = [0.51 * dx**2, 0.49 * dx**2, 0.25 * dx**2]

# Definim els paràmetres normalitzats
x0 = 0.02
t0 = 0.02**2/alpha
tf = 0.025

# Definim les temperatures normalitzades
T0 = (0.472*(40/np.sqrt(2))**2)/0.56
Tc = (36.50 + 273.15)/T0

# Definim els punts espaials
x = np.linspace(0, 1, Nx)

# Definim la solució analítica

def T_analitica(x, t, Tc):
    T = Tc * np.ones(len(x))

    for n in range(1, 101):
        T += (4 / ((2*n-1)**3 * np.pi**3)) * (1 - np.exp(-(2*n-1)**2 * np.pi**2 * t)) * np.sin((2*n-1) * np.pi * x)

    return T


# Llistes per guardar els resultats

T_f = []
errors = []

# Resolem numèricament l'equació

# Introduïm l'expressió de l'algorisme trobada pel mètode d'Euler explícit.

for s in dt:
    gamma = s/(dx)**2
    Nt = round(tf/s, None)         # Nombre de passes temporals

    # Condició inicial: T(x, 0) = Tc a tot el domini
    T = Tc * np.ones(Nx)

    # Bucle temporal 
    for i in range(Nt):

        T_nova = np.array(T)

        for j in range(1, Nx-1):
            T_nova[j] = T[j] + gamma * (T[j+1] - 2*T[j] + T[j-1]) + s
            
        T = np.array(T_nova)

    # Guardem el resultat final

    T_f.append(T)

    # Calculem la solució analítica al mateix temps
    T_exacta = T_analitica(x, Nt*s, Tc)

    # Calculem l'error
    error = np.abs(T - T_exacta)

    errors.append(error)


# Representació gràfica de les resolucions numèriques

plt.figure(figsize=(7,5))

plt.plot(x, T_f[0], label='a = 0.51', color = 'royalblue')
plt.plot(x, T_f[1], label='a = 0.49', color = 'red')
plt.plot(x, T_f[2], label='a = 0.25', color = 'limegreen')

plt.xlabel('x normalitzada')
plt.ylabel('T normalitzada')
plt.title('Distribució de temperatura en funció de la posició')
plt.legend()
plt.grid()
plt.show()


# Comparació amb la solució analítica

# Utilitzem el temps final de la simulació a = 0.49
T_exacta = T_analitica(x, round(tf/dt[1])*dt[1], Tc)

plt.figure(figsize=(7,5))

plt.plot(x, T_f[1], label='Solució numèrica a = 0.49', color = 'royalblue')
plt.plot(x, T_f[2], label='Solució numèrica a = 0.25', color = 'limegreen')
plt.plot(x, T_exacta, linestyle = '--', label='Solució analítica', color = 'darkorange')

plt.xlabel('x normalitzada')
plt.ylabel('T normalitzada')
plt.title('Comparació de les solucions')
plt.legend()
plt.grid()
plt.show()


# Representació gràfica dels errors

plt.figure(figsize=(7,5))

# plt.plot(x, errors[0], label='Error a = 0.51', color = 'green') 
plt.plot(x, errors[1], label='Error a = 0.49', color = 'royalblue')
plt.plot(x, errors[2], label='Error a = 0.25', color = 'red')

plt.xlabel('x normalitzada')
plt.ylabel('Error absolut')
plt.title('Errors numèrics per a = 0.25 i a = 0.49')
plt.legend()
plt.grid()
plt.show()
