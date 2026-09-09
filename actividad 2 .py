import numpy as np
from numpy import sin, cos
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle

#----------------------------------------------------
# System parameters
#----------------------------------------------------
g = 9.81
l = 1.0      # longitud del péndulo
m = 1.0      # masa del péndulo
M = 4.0      # masa del bloque
k = 50.0     # constante elástica del resorte

#----------------------------------------------------
# Initial conditions
#----------------------------------------------------
x0  = 0.2                # posición inicial del bloque (desplazamiento del resorte)
v0  = 0.0                # velocidad inicial del bloque
th0 = np.radians(20.0)   # ángulo inicial del péndulo (desde la vertical)
w0  = 0.0                # velocidad angular inicial del péndulo

#----------------------------------------------------
# method parameters
#----------------------------------------------------
tmax = 20
dt = 0.01
STRIDE = 2

#----------------------------------------------------
# Dynamics: Pendulum on a sliding support attached to a spring-block
#----------------------------------------------------
def dyn(t, y):
    x, v, th, w = y
    s = sin(th)
    c = cos(th)
    D = M/m + s**2
    vdot = ((g*c + l*w**2)*s - (k/m)*x) / D
    wdot = -(1/l) * (g*(1 + M/m)*s + c*(l*w**2*s - (k/m)*x)/D)
    return np.array([v, vdot, w, wdot])

# ----------------------------------------------------------
# Fourth-order Runge-Kutta method
# ----------------------------------------------------------
def rk4(f, t, y, h):
    k1 = h * f(t, y)
    k2 = h * f(t + h/2, y + k1/2)
    k3 = h * f(t + h/2, y + k2/2)
    k4 = h * f(t + h, y + k3)
    return y + (k1 + 2 * k2 + 2 * k3 + k4) / 6

# ----------------------------------------------------------
# Integration using RK4
# ----------------------------------------------------------
n = int(tmax / dt)
t = np.linspace(0, n * dt, n + 1)
y = np.empty((n + 1, 4))
y[0] = np.array([x0, v0, th0, w0])
for i in range(n):
    y[i + 1] = rk4(dyn, t[i], y[i], dt)

# ----------------------------------------------------------
# Separate variables after integration
# ----------------------------------------------------------
x  = y[:, 0]
v  = y[:, 1]
th = y[:, 2]
w  = y[:, 3]

# ----------------------------------------------------------
# Kinematics
# ----------------------------------------------------------
BLOCK_W, BLOCK_H = 0.35, 0.30
# El bloque se desliza sobre el eje horizontal (y = 0)
xb, yb = x, np.zeros_like(x)
# La cuerda del péndulo sale siempre del borde inferior del bloque
y_pivot = -BLOCK_H / 2
# La masa pendular cuelga de ese borde, a distancia l
xp, yp = xb + l * sin(th), y_pivot - l * cos(th)

# ----------------------------------------------------------
# Figure
# ----------------------------------------------------------
X_WALL = -2.0  # posición fija de la pared (anclaje del resorte)
L_REF = abs(X_WALL)  # longitud de referencia para escalar la amplitud del resorte

# Paleta de colores
COLOR_BLOCK  = "#7B52AB"   # morado del bloque
COLOR_MASS   = "#6FA8DC"   # azul claro de la masa pendular
COLOR_ROD    = "#000000"   # varilla del péndulo (negro)
COLOR_SPRING = "#333333"   # gris oscuro/negro del resorte
COLOR_TRACE  = "#D90BDC"   # azul claro de la traza
COLOR_WALL   = "#000000"   # borde de la pared (negro)


def spring_xy(x_start, x_end, n_coils=14):
    """Genera las coordenadas de un resorte helicoidal entre x_start y x_end,
    con espiras más densas cerca de los extremos y amplitud que se ajusta
    según qué tan estirado/comprimido está (más realista que un zigzag fijo)."""
    L = abs(x_end - x_start) + 1e-9
    s = np.linspace(0, 1, 300)
    s_nonuniform = 0.5 * (1 - np.cos(np.pi * s))  # densidad no uniforme
    amp = np.clip(0.12 * np.sqrt(L_REF / L), 0.05, 0.14)
    xs = x_start + (x_end - x_start) * s_nonuniform
    ys = amp * np.sin(2 * np.pi * n_coils * s_nonuniform)
    return xs, ys

plt.style.use("seaborn-v0_8-whitegrid")
fig, ax = plt.subplots(figsize=(9, 6))
fig.patch.set_facecolor("white")

# Límites de la gráfica calculados a partir del movimiento real
# (si cambias l, k, M, m o las condiciones iniciales, el encuadre se
# ajusta solo en vez de quedarse fijo con los valores originales)
PAD = 0.4
x_min = min(X_WALL - 0.3, xp.min() - PAD, (xb - BLOCK_W/2).min() - PAD)
x_max = max(xp.max() + PAD, (xb + BLOCK_W/2).max() + PAD)
y_min = min(yp.min() - PAD, -BLOCK_H - PAD)
y_max = max(yb.max() + BLOCK_H + PAD, 0.6)

ax.set(xlim=(x_min, x_max), ylim=(y_min, y_max), aspect="equal",
       title="Péndulo con soporte sobre bloque-resorte (RK4)")
ax.title.set_fontsize(20)
ax.title.set_fontweight("bold")
ax.tick_params(axis="both", labelsize=14)
ax.grid(alpha=0.3, lw=0.8)

# Líneas guía: nivel del riel y posición de equilibrio del bloque
ax.axhline(0, color="gray", lw=0.8, alpha=0.4, zorder=0)
ax.axvline(0, color="gray", lw=0.8, ls="--", alpha=0.4, zorder=0)

# Pared fija, con textura de anclaje (no cambia con el tiempo)
wall = Rectangle((X_WALL - 0.15, -0.35), 0.15, 0.7,
                  facecolor="#dddddd", edgecolor=COLOR_WALL,
                  hatch="////", lw=1.2, zorder=5)
ax.add_patch(wall)

spring, = ax.plot([], [], "-", lw=2, color=COLOR_SPRING,
                   solid_capstyle="round", zorder=2)
block = Rectangle((xb[0] - BLOCK_W/2, -BLOCK_H/2), BLOCK_W, BLOCK_H,
                   facecolor=COLOR_BLOCK, edgecolor="black", lw=1.3, zorder=4)
ax.add_patch(block)
rod,    = ax.plot([], [], "-", lw=2, color=COLOR_ROD, zorder=3)
mass,   = ax.plot([], [], "o", markersize=20, color=COLOR_MASS,
                   markeredgecolor="black", markeredgewidth=1.2, zorder=6)
trace,  = ax.plot([], [], "-", lw=1.2, alpha=0.5, color=COLOR_TRACE, zorder=1)
clock   = ax.text(0.03, 0.93, "", transform=ax.transAxes, fontsize=16,
                   bbox=dict(boxstyle="round", facecolor="white",
                             alpha=0.7, edgecolor="gray"))

# ----------------------------------------------------------
# Create Animation
# ----------------------------------------------------------
def animate(i):
    left_edge = xb[i] - BLOCK_W / 2
    xs, ys = spring_xy(X_WALL, left_edge)
    spring.set_data(xs, ys)
    block.set_xy((left_edge, -BLOCK_H / 2))
    rod.set_data([xb[i], xp[i]], [y_pivot, yp[i]])
    mass.set_data([xp[i]], [yp[i]])
    trace.set_data(xp[:i + 1], yp[:i + 1])
    clock.set_text(f"t = {t[i]:.1f} s")
    return spring, block, rod, mass, trace, clock

# ----------------------------------------------------------
# Animation
# ----------------------------------------------------------
ani = FuncAnimation(fig, animate, frames=range(0, n + 1, STRIDE),
                     interval=STRIDE * dt * 1000, blit=False)
plt.tight_layout()
plt.show()