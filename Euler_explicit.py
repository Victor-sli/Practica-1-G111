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

# Coordenades de la zona malalta 
x1 = 0.0075
x2 = 0.0125

# Coordenades de la zona malalta normalitzades
x1n = x1/x0
x2n = x2/x0

# Definim les temperatures normalitzades
T0 = (0.472*(40/np.sqrt(2))**2)/0.56
Tc = (36.50 + 273.15)/T0
T50 = (50 + 273.15) / T0
T80 = (80 + 273.15) / T0

# Definim els punts espaials
x = np.linspace(0, 1, Nx)

# Definim la regió malalta
regio_malalta = (x >= 0.375) & (x <= 0.625)

# Definim la regió sana
regio_sana = (x < 0.375) | (x > 0.625)


# Definim la solució analítica

def T_analitica(x, t, Tc):
    T = Tc * np.ones(len(x))

    for n in range(1, 101):
        bn = 4 / ((2*n-1) * np.pi)
        
        T += bn * ((1 - np.exp(-(2*n-1)**2 * np.pi**2 * t))/ ((2*n-1)**2 * np.pi**2)) * np.sin((2*n-1) * np.pi * x)
        
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


# Utilitzem el pas temporal estable a = 0.49 (amb dt[1]). També es pot fer servir a = 0.25 (amb dt[2])
s = dt[1]
gamma = s / dx**2

# Definim la temperatura inicial
T = Tc * np.ones(Nx)

# Temps inicial
temps = 0

# Simulació fins que s'arribi a una condició de seguretat
while True:

    # Temperatura màxima de la regió sana
    Tmax_sana = np.max(T[regio_sana])

    # Temperatura màxima de tot el teixit
    T_max = np.max(T)

    # Comprovem les condicions de seguretat
    if Tmax_sana >= T50 or T_max >= T80:
        break

    # Actualitzem la temperatura
    T_nova = T.copy()

    for j in range(1, Nx-1):
        T_nova[j] = T[j] + gamma * (T[j+1] - 2*T[j] + T[j-1]) + s

    T = T_nova

    # Actualitzem el temps
    temps += s

# Convertim el temps normalitzat a segons
temps_segons = temps * t0

print("Temps del tractament t =", temps_segons, "s")



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
plt.title('Comparació de les solucions numèriques i analítica')
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
