import numpy as np
import matplotlib.pyplot as plt


# comparar geoemtria 



# Punto inicial
xn = 0.5
yn = 0.0

# Número máximo de iteraciones
N = 2000000
x = np.zeros(N)
y = np.zeros(N)

# Iteraciones del sistema de funciones iteradas
for n in range(N):

    r = np.random.random()

    if r < 0.10:
        xn, yn = 0.05 * xn, 0.6 * yn

    elif r < 0.20:
        xn, yn = 0.05 * xn, -0.5 * yn + 1

    elif r < 0.40:
        xn, yn = 0.46 * xn - 0.15 * yn, 0.39 * xn + 0.38 * yn + 0.6

    elif r < 0.60:
        xn, yn = 0.47 * xn - 0.15 * yn, 0.17 * xn + 0.42 * yn + 1.1

    elif r < 0.80:
        xn, yn = 0.43 * xn + 0.28 * yn, -0.25 * xn + 0.45 * yn + 1

    else:
        xn, yn = 0.42 * xn + 0.26 * yn, -0.35 * xn + 0.31 * yn + 0.7

    x[n] = xn
    y[n] = yn

# Distintos números de iteraciones a comparar
Ns = [1000, 10000, 100000, 2000000]
tamanos = [2, 0.5, 0.05, 0.001]   # tamaño del punto para cada N

# Gráfica
fig, axs = plt.subplots(1, 4, figsize=(16, 7))

for ax, Ni, s in zip(axs, Ns, tamanos):
    ax.scatter(x[:Ni], y[:Ni], s=s, color='deeppink')
    ax.set_title(f'N = {Ni:,}', fontsize=14)
    ax.tick_params(axis='both', labelsize=10)
    ax.set_aspect('equal')

plt.tight_layout()

plt.savefig("ocobo_comparacion.png", format="png", bbox_inches="tight", dpi=400)
plt.show()

