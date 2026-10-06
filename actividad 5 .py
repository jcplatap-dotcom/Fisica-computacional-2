
import numpy as np
from numpy import sin,cos,pi
import matplotlib.pyplot as plt 

# parametros del sistema

delta = 0.1
alpha = 2.0
beta  = 2.0
om    = 1.2
T     = 2*pi/om

# condiciones iniciales

x0, v0, = 1, 1

# parametros de bifurcacion: gamma

gamma_min = 0.1
gamma_max = 7.0
dgamma = 0.001
gamma_values = np.arange(gamma_min, gamma_max +  dgamma, dgamma)
n_orbits = len(gamma_values)

# estado inicial 
x= np.full(n_orbits, x0)
v= np.full(n_orbits, v0)
y= np.concatenate([x, v])

# parametros del metodo numerico 

#rans, Nkeep, steps_per_T, dt = 350, 200, 300, T / steps_per_T

Trans=250
Nkeep=100
steps_per_T=250
dt=T/steps_per_T

# dimamica 
def dyn (t,y):
    x = y[:n_orbits]
    v = y[n_orbits:]
    dx = v
    dv = -delta*v + alpha*x - beta*x**3 + gamma_values*cos(om*t)
    return np.concatenate([dx, dv])

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

fig, ax = plt.subplots (figsize=(8,6))
for i in range(n_orbits):
    ax.scatter (np.full(Nkeep, gamma_values[i]),
                v_strobe[:, i], s=0.5, color='blue', linewidths=0, rasterized=True) 
    # se pone en v_strobe, x_strobe para observar la grafica de x en funcion de lamda
ax.set_xlabel(r'$\gamma$', fontsize= 16)
ax.set_ylabel(r'$\dot{x}$', fontsize=16)
ax.tick_params(axis='both', labelsize=12)
ax.set_xlim(gamma_min, gamma_max) 
ax.set_box_aspect(0.65)
plt.tight_layout()
plt.savefig('Bifurcacion v.pdf', format='pdf', bbox_inches='tight', dpi=800)
plt.show()
                
    
      
    
