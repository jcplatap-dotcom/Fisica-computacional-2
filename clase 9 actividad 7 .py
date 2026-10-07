import numpy as np
from math import sin, cos, pi
import matplotlib.pyplot as plt

# definimos los parametros del sistema

q = 2.0
gamma = 1.1799
wD = 2.0/3.0          # NO aparece en la imagen: valor asumido, cámbialo si es otro
T = 2*pi/wD           # periodo de la fuerza (muestreo estroboscópico)


# Condición inicial (theta, theta_punto)

x0, v0 = 1.0, 1.0


# Parámetros 

Trans = 300           # periodos descartados (transitorio)
Nkeep = 10000         # periodos guardados (puntos del mapa)
steps_per_T = 200
dt = T/steps_per_T


# Dinámica: theta'' = -(1/q) theta' - sin(theta) + gamma cos(wD t)

def dyn(t, x, v):
    dx = v
    dv = -v/q - sin(x) + gamma*cos(wD*t)
    return dx, dv


# Runge-Kutta de cuarto orden

def rk4(t, x, v, h):
    k1x, k1v = dyn(t, x, v)
    k2x, k2v = dyn(t + h/2, x + h*k1x/2, v + h*k1v/2)
    k3x, k3v = dyn(t + h/2, x + h*k2x/2, v + h*k2v/2)
    k4x, k4v = dyn(t + h, x + h*k3x, v + h*k3v)
    x_new = x + h*(k1x + 2*k2x + 2*k3x + k4x)/6
    v_new = v + h*(k1v + 2*k2v + 2*k3v + k4v)/6
    return x_new, v_new

# Almacenamiento estroboscópico

theta_strobe = np.empty(Nkeep)
v_strobe = np.empty(Nkeep)
save_index = 0


# Integración

x, v = x0, v0
total_steps = (Trans + Nkeep)*steps_per_T
for step in range(total_steps):
    t = step*dt
    x, v = rk4(t, x, v, dt)
    if (step + 1) % steps_per_T == 0:
        if (step + 1)//steps_per_T > Trans:
            # theta envuelto a [-pi, pi)
            theta_strobe[save_index] = (x + pi) % (2*pi) - pi
            v_strobe[save_index] = v
            save_index += 1


# Mapa estroboscópico: theta_punto vs theta


fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(theta_strobe, v_strobe, s=0.5, color='crimson', linewidths=0)

ax.set_xlabel(r'$\theta$', fontsize=16)
ax.set_ylabel(r'$\dot{\theta}$', fontsize=16)
#ax.set_title(rf'Mapa estroboscópico: $\gamma={gamma}$, $q={q:g}$', fontsize=14)
ax.tick_params(axis='both', labelsize=12)
ax.text(0.03, 0.97, rf'$q={q:g}$', transform=ax.transAxes,
        ha='left', va='top', fontsize=12)
ax.text(0.97, 0.97, rf'$\gamma={gamma}$', transform=ax.transAxes,
        ha='right', va='top', fontsize=12)
ax.text(0.03, 0.06, rf'$\omega=2/3$', transform=ax.transAxes,
        ha='left', va='top', fontsize=12)
plt.tight_layout()
plt.savefig("mapa_estroboscopico actividad 7.pdf", dpi=300)
plt.show()