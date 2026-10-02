import numpy as np
from numpy import cos, pi
import matplotlib.pyplot as plt

# cambiamos los parametros del ejercicio
delta = 0.15
alpha = 1.0
beta  = 1.0
gamma = 0.3
om    = 1
T = 2*pi / om          # periodo de la fuerza = 4*pi

# condiciones iniciales
rng = np.random.default_rng(0)

n_orbits = 80

x0_flat = rng.uniform(-2, 2, n_orbits)
v0_flat = rng.uniform(-2, 2, n_orbits)

y = np.concatenate([x0_flat, v0_flat])

# parametros del metodo
Trans = 200            # periodos descartados (transitorio)
Nperiods = 2500        # periodos totales
steps_per_T = 200      # pasos de RK4 por periodo
dt = T / steps_per_T

# -----------------------------------------------------
# Dinámica: oscilador de Duffing forzado
# x'' = gamma*cos(om*t) - delta*x' + alpha*x - beta*x^3
# -----------------------------------------------------
def dyn(t, y):
    x = y[:n_orbits]
    v = y[n_orbits:]
    dx = v
    dv = gamma*cos(om*t) - delta*v + alpha*x - beta*x**3
    return np.concatenate([dx, dv])

# -----------------------------------------------------
# Método de Runge-Kutta de cuarto orden
# -----------------------------------------------------
def rk4(f, t, y, h):
    k1 = h * f(t, y)
    k2 = h * f(t + h/2, y + k1/2)
    k3 = h * f(t + h/2, y + k2/2)
    k4 = h * f(t + h, y + k3)
    return y + (k1 + 2*k2 + 2*k3 + k4) / 6

# -----------------------------------------------------
# Almacenamiento estroboscópico
# -----------------------------------------------------
n_saved = Nperiods - Trans + 1
x_strobe = np.empty((n_saved, n_orbits))
v_strobe = np.empty((n_saved, n_orbits))

save_index = 0
if Trans == 0:
    x_strobe[save_index] = y[:n_orbits]
    v_strobe[save_index] = y[n_orbits:]
    save_index += 1

# -----------------------------------------------------
# Integración
# -----------------------------------------------------
total_steps = Nperiods * steps_per_T
for step in range(total_steps):
    current_time = step*dt
    y = rk4(dyn, current_time, y, dt)
    completed_period = (step + 1) // steps_per_T
    if (step + 1) % steps_per_T == 0:
        if completed_period >= max(1, Trans):
            x_strobe[save_index] = y[:n_orbits]
            v_strobe[save_index] = y[n_orbits:]
            save_index += 1


# -----------------------------------------------------
# Mapa estroboscópico (color según condición inicial)
# -----------------------------------------------------
colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']
orbit_colors = [colors[i % len(colors)] for i in range(n_orbits)]
fig, ax = plt.subplots(figsize=(8, 6))
for i in range(n_orbits):
    ax.scatter(x_strobe[:, i], v_strobe[:, i], s=1, color=orbit_colors[i],
               linewidths=0, rasterized=True)
ax.set_xlabel(r"$x$", fontsize=22)
ax.set_ylabel(r"$\dot{x}$", fontsize=22)
ax.tick_params(axis="both", labelsize=18)
ax.set_box_aspect(0.65)
plt.tight_layout()
plt.savefig('graf 1,b.pdf', format='pdf', bbox_inches='tight', dpi=800)
plt.show()