
import numpy as np
from numpy import sin,cos,pi
import matplotlib.pyplot as plt 
from fractions import Fraction

# definimos los nuevos parametros del sistema

alpha = 0.1
omega0 = 1.0
om = 2.0

T = 2*pi/om

# condiciones iniciales

theta = 1
thetapunto = 1

# parametros de bifurcacion: gamma

gamma_min = 0
gamma_max = 2.25
dgamma = 0.001
gamma_values = np.arange(gamma_min, gamma_max +  dgamma, dgamma)
n_orbits = len(gamma_values)

# estado inicial 
theta = np.full(n_orbits, theta)
v = np.full(n_orbits, thetapunto)

y = np.concatenate([theta, v])

# parametros del metodo numerico 

#rans, Nkeep, steps_per_T, dt = 350, 200, 300, T / steps_per_T

Trans=300
Nkeep=350
steps_per_T=250
dt=T/steps_per_T


# dimamica 

def dyn(t, y):
    theta = y[:n_orbits]
    v = y[n_orbits:]

    dtheta = v

    dv = (-alpha*v - omega0**2 * sin(theta) + gamma_values * cos(om*t) * sin(theta))

    return np.concatenate([dtheta, dv])


# metodo runge_kutta
def rk4( f, t, y, h):
    k1 = h*f(t,y)
    k2 = h*f(t + h/2, y + k1/2)
    k3 = h*f(t + h/2, y + k2/2)
    k4 = h*f(t + h/2, y + k3)
    return y + (k1 + 2*k2 + 2*k3 + k4)/6

# estroboscopico 

x_strobe = np.empty ((Nkeep, n_orbits))
v_strobe = np.empty ((Nkeep, n_orbits))
save_index = 0

# integracion 
total_periods = Trans + Nkeep
total_steps = total_periods * steps_per_T
for step in range(total_steps):
    current_time = step*dt
    y = rk4 (dyn, current_time, y, dt)
    completed_period = (step + 1) // steps_per_T
    if (step + 1 ) % steps_per_T == 0:
        if completed_period > Trans:
            x_strobe[save_index] = y[:n_orbits]
            v_strobe[save_index] = y[n_orbits:]
            save_index += 1
        
# diagrama de bifurcacion y formato de figura 

fig, ax = plt.subplots(figsize=(8, 6))

for i in range(n_orbits):

    ax.scatter(
        np.full(Nkeep, gamma_values[i]),
        np.abs(v_strobe[:, i]),
        s=0.5,
        color='blue',
        linewidths=0,
        rasterized=True
    )

ax.set_xlabel(r'$\gamma$', fontsize=16)
ax.set_ylabel(r'$|\dot{\theta}|$', fontsize=16)

ax.tick_params(axis='both', labelsize=12)

ax.set_xlim(gamma_min, gamma_max)

ax.set_box_aspect(0.65)

plt.tight_layout()

plt.savefig(
    'Bifurcation.pdf',
    format='pdf',
    bbox_inches='tight',
    dpi=800
)

plt.show()
                
    
      
    
