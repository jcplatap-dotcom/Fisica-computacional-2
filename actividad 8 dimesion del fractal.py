import numpy as np
import matplotlib.pyplot as plt

# ---------- 1) Generar el fractal ----------
P = np.array([0.5, 0])
N = 2000000
x = np.zeros(N)
y = np.zeros(N)

for n in range(N):
    r = np.random.random()
    xn = P[0]
    yn = P[1]

    if r < 0.10:
        P = np.array([0.05*xn, 0.6*yn])
    elif r < 0.20:
        P = np.array([0.05*xn, -0.5*yn + 1])
    elif r < 0.40:
        P = np.array([0.46*xn - 0.15*yn, 0.39*xn + 0.38*yn + 0.6])
    elif r < 0.60:
        P = np.array([0.47*xn - 0.15*yn, 0.17*xn + 0.42*yn + 1.1])
    elif r < 0.80:
        P = np.array([0.43*xn + 0.28*yn, -0.25*xn + 0.45*yn + 1])
    else:
        P = np.array([0.42*xn + 0.26*yn, -0.35*xn + 0.31*yn + 0.7])

    x[n] = P[0]
    y[n] = P[1]

# ---------- 2) Normalizar al cuadrado mínimo ----------
xmin, xmax = x.min(), x.max()
ymin, ymax = y.min(), y.max()
L  = max(xmax - xmin, ymax - ymin)
x0 = (xmin + xmax)/2 - L/2
y0 = (ymin + ymax)/2 - L/2
xs = (x - x0)/L
ys = (y - y0)/L

# ---------- 3) Conteo de cajas ----------
ks = np.arange(2, 17)
conteos = []
for k in ks:
    n  = 2**k
    ix = np.minimum((xs*n).astype(np.int64), n - 1)
    iy = np.minimum((ys*n).astype(np.int64), n - 1)
    conteos.append(len(np.unique(ix*n + iy)))
conteos = np.array(conteos)

ln_inv_eps = ks*np.log(2)
ln_N = np.log(conteos)

# ---------- 4) Tabla de pendientes locales ----------
pend = np.diff(ln_N)/np.diff(ln_inv_eps)
print(" k      n      N(eps)   pend. local")
for i, k in enumerate(ks):
    p = f"{pend[i]:.3f}" if i < len(pend) else "-"
    print(f"{k:2d} {2**k:6d} {conteos[i]:10d}   {p}")

# ---------- 5) UNA figura con cuadrícula ----------
k_fig = 7                     
n  = 2**k_fig
ix = np.minimum((xs*n).astype(np.int64), n - 1)
iy = np.minimum((ys*n).astype(np.int64), n - 1)
ocupada = np.zeros((n, n))
ocupada[iy, ix] = 1
ocupada = np.ma.masked_equal(ocupada, 0)

fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(ocupada, origin='lower', extent=[0, 1, 0, 1],
          cmap='Blues', vmin=0, vmax=4, interpolation='nearest')
bordes = np.linspace(0, 1, n + 1)
ax.vlines(bordes, 0, 1, color='gray', linewidth=0.3)
ax.hlines(bordes, 0, 1, color='gray', linewidth=0.3)
ax.scatter(xs[::10], ys[::10], s=0.1, color='deeppink', linewidths=0)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title(rf'$n={n},\quad N(\varepsilon)={conteos[k_fig-2]}$', fontsize=12)
plt.tight_layout()
plt.savefig("ocobo_cuadricula.png", dpi=400, bbox_inches="tight")
plt.show()

# ---------- 6) Ajuste lineal
k_min, k_max = 2, 9
mask = (ks >= k_min) & (ks <= k_max)
Df, b = np.polyfit(ln_inv_eps[mask], ln_N[mask], 1)
print(f"D_f = {Df:.4f}")

plt.figure(figsize=(6, 4.5))
plt.plot(ln_inv_eps, ln_N, 'o', markersize=4, color='gray')
plt.plot(ln_inv_eps[mask], ln_N[mask], 'o', markersize=4, color='deeppink')
plt.plot(ln_inv_eps[mask], Df*ln_inv_eps[mask] + b, 'k-',
         label=rf'$D_f = {Df:.4f}$')
plt.xlabel(r'$\ln(1/\varepsilon)$', fontsize=14)
plt.ylabel(r'$\ln N(\varepsilon)$', fontsize=14)
plt.legend()
plt.tight_layout()
plt.savefig("ocobo_regresion.png", dpi=300, bbox_inches="tight")
plt.show()