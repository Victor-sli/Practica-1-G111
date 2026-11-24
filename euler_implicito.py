import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3})

# ---------- Constants i dades de l'enunciat ----------
cv    = 3698      # calor especific      [J/(kg K)]
rho   = 1081      # densitat              [kg/m^3]
kappa = 0.56      # conductivitat termica [W/(m K)]
sigma = 0.472     # conductivitat electrica [S/m]

# ---------- Geometria y excitacio ----------
d      = 0.02                 # distancia entre electrodes = gruixut de la paret [m]
V_pic  = 40.0                 # potencial de pic [V]
V_ef   = V_pic / np.sqrt(2)   # potencial efectiu [V]
Tc     = 36.5                 # temperatura corporal [C]

# ---------- Adimensionalitzacio ----------
alpha = kappa / (cv * rho)          # difusivitat termica [m^2/s]
x0    = d                           # escala de longitud [m]
t0    = d**2 / alpha                # escala de temps [s]
T0    = sigma * V_ef**2 / kappa     # escala de temperatura [K]

beta  = Tc / T0                     # condicio inicial y de contorn adimensional

print(f"alpha = {alpha:.4e} m^2/s")
print(f"x0 = {x0} m,  t0 = {t0:.3f} s,  T0 = {T0:.3f} K")
print(f"beta = Tc/T0 = {beta:.5f}")

# ---------- Adimensionalitzacio de llindars clinics ----------
T_muerte = 50.0 / T0     # punt des d'on hi ha mort cel·lular
T_max    = 80.0 / T0     # punt on comença a formar-se trombosi

# ---------- Discretitzacio ----------
N  = 101                 # nodes espacials
ta = 0.025               # temps final adimensional
dx = 1.0 / (N - 1)       # pas espacial (mida adimensional de la paret = 1)

print(f"dx = {dx}, ta = {ta}  ->  t fisico = {ta*t0:.1f} s")

def T_analitica(x, t, beta, n_terminos=500):
    # Suma de la serie del anexo, sólo con los términos impares n = 1, 3, 5, ...
    n    = np.arange(1, 2 * n_terminos, 2)                       # n impares
    bn   = 4.0 / (n * np.pi)                                     # coeficientes b_n de q = 1
    coef = bn * (1 - np.exp(-n**2 * np.pi**2 * t)) / (n**2 * np.pi**2)
    # matriz (N x n_terminos) de senos, multiplicada por los coeficientes
    return beta + np.sin(np.outer(x, n * np.pi)) @ coef

# Comprobación rápida: en t = 0 debe dar beta en todos los puntos
x_test = np.linspace(0, 1, N)
print("max|T(x,0) - beta| =", np.max(np.abs(T_analitica(x_test, 0.0, beta) - beta)))

def gauss_seidel(T_old, T, r, dt, tol, max_iter, hist):
    """
    Resuelve UN paso temporal de Euler implícito.

    T_old : temperatura en el instante n (fija; forma parte del término independiente b)
    T     : entra con el "guess" inicial y sale con la solución en n+1 (se modifica in situ)
    hist  : vector donde se guarda max|T^(k+1)-T^(k)| de cada iteración (para analizar convergencia)

    Devuelve el número de iteraciones usadas, o -1 si no convergió en max_iter.
    """
    N = T.shape[0]
    for k in range(max_iter):
        max_dif = 0.0
        for i in range(1, N - 1):              # sólo nodos interiores; T[0] y T[N-1] son el contorno
            nuevo = (T_old[i] + dt + r * (T[i - 1] + T[i + 1])) / (1.0 + 2.0 * r)
            dif = abs(nuevo - T[i])
            if dif > max_dif:
                max_dif = dif
            T[i] = nuevo                       # se sobrescribe YA: el siguiente nodo usará este valor nuevo
        hist[k] = max_dif
        if max_dif < tol:                      # criterio de parada: el barrido ya casi no cambia nada
            return k + 1
    return -1

def euler_implicito(N, r, ta, beta, tol=1e-12, max_iter=100000):
    """
    Integra desde t = 0 hasta t = ta con paso dt = r * dx^2.
    Devuelve x, T(x, ta) y el número de iteraciones de GS que se usaron en cada paso temporal.
    """
    dx = 1.0 / (N - 1)
    dt = r * dx**2
    n_pasos = int(round(ta / dt))
    assert abs(n_pasos * dt - ta) < 1e-12, "ta / dt no es entero: no se llega exactamente a ta"

    x = np.linspace(0.0, 1.0, N)
    T = np.full(N, beta)                       # condición inicial Y de contorno: todo a beta
    hist = np.zeros(max_iter)
    iters = np.zeros(n_pasos, dtype=int)

    for n in range(n_pasos):                   # bucle temporal
        T_old = T.copy()                       # T^n, congelado durante todo el paso
        # T entra como guess (= T^n, buen punto de partida porque cambia poco entre pasos)
        k = gauss_seidel(T_old, T, r, dt, tol, max_iter, hist)
        if k < 0:
            raise RuntimeError(f"Gauss-Seidel no convergió en el paso {n}")
        iters[n] = k
    return x, T, iters

# ---------- Paso 1: elegir parámetros ----------
# dt = r * dx^2, así que dar r equivale a dar dt. Número de pasos: 250 para r = 1, 500 para r = 0.5.
N   = 101
tol = 1e-12                    # claramente por debajo del error de truncamiento (ver sección 3b)
casos = {r"$\Delta t=\Delta x^2$": 1.0,
         r"$\Delta t=0.5\,\Delta x^2$": 0.5}

# ---------- Integracio en el temps y calcul de l'error numeric ----------

resultados = {}
for etiqueta, r in casos.items():
    x, T_num, iters = euler_implicito(N, r, ta, beta, tol)
    T_an  = T_analitica(x, ta, beta)           # solucio exacta al mateix instant ta
    error = np.abs(T_num - T_an)               # error punt a punt
    resultados[etiqueta] = (T_num, T_an, error, iters)
    print(f"r = {r:<4} -> pasos = {len(iters):>3},  iter. GS/paso (media) = {iters.mean():.1f},  "
          f"error máx = {error.max():.3e}")

# ---------- Gráficas ----------
x_cm = x * x0 * 100     # x adimensional -> cm

fig, axs = plt.subplots(1, 2, figsize=(16, 4.5))

# (a) Perfil de temperatura en t = ta
T_an_fis = T_analitica(x, ta, beta) * T0       # T[°C] = T_adim * T0 (porque beta se definió con Tc en °C)
axs[0].plot(x_cm, T_an_fis, "k-", lw=1.5, label="analítica")
for (etiqueta, (T_num, _, _, _)), marcador in zip(resultados.items(), ["o", "s"]):
    axs[0].plot(x_cm[::4], T_num[::4] * T0, marcador, ms=4, mfc="none", label=etiqueta)
axs[0].axhline(50, color="tab:orange", ls="--", label="50 °C (mort cel·lular)")
axs[0].axhline(80, color="tab:red", ls="--", label="80 °C (límit)")
axs[0].axvspan(0.75, 1.25, color="tab:blue", alpha=0.12, label="teixit malalt")
axs[0].set_xlabel("x [cm]"); axs[0].set_ylabel("T [°C]")
axs[0].set_title(f"Perfil en t = {ta*t0:.1f} s"); axs[0].legend(fontsize=8)

# (b) Error numérico punto a punto
for etiqueta, (_, _, error, _) in resultados.items():
    axs[1].plot(x_cm, error, label=etiqueta)
axs[1].set_xlabel("x [cm]"); axs[1].set_ylabel(r"$|T^{num}-T^{an}|$ (adimensional)")
axs[1].set_title("Error numérico"); axs[1].legend(fontsize=8)
axs[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

plt.tight_layout()
plt.show()