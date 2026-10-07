import numpy as np
from numpy import sin, cos, pi, log
import matplotlib.pyplot as plt
# calcular el jacobiano del sistema
# parametros del sistema

q = 2
om = 2/3
T = 2*pi / om

# condiciones iniciales

x0 = 1.25
v0 = 1
t = 0

# valores de gamma

gamma_min = 0.9
gamma_max = 1.8
dgamma = 0.0001
gamma_values = np.arange(gamma_min, gamma_max +  dgamma, dgamma)
n_orbits = len(gamma_values)

# parametros del metodo 

nTrans = 300
nLyap = 600 
steps_per_T = 300
dt = T / steps_per_T

# estado incial y vector unitario tangente 

x = np.full(n_orbits, x0)
v = np.full(n_orbits, v0)
xi = np.full(n_orbits, 1.0/np.sqrt(2.0))
eta = np.full(n_orbits, 1.0/np.sqrt(2.0))
y = np.concatenate ((x, v, xi, eta))

# dinamica 


def dyn(t, y):

    x = y[:n_orbits]
    v = y[n_orbits:2*n_orbits]
    xi = y[2*n_orbits:3*n_orbits]
    eta = y[3*n_orbits:4*n_orbits]

    dx = v
    dv = -(1/q)*v - sin(x) + gamma_values*cos(om*t)

    dxi = eta
    deta = -cos(x)*xi - (1/q)*eta

    return np.concatenate([dx, dv, dxi, deta])

def rk4(f, t, y, h):
    k1 = h*f(t, y)
    k2 = h*f(t + h/2, y + k1/2)
    k3 = h*f(t + h/2, y + k2/2)
    k4 = h*f(t + h, y + k3)
    return y + (k1 + 2*k2 + 2*k3 + k4)/6

# integracion 

for period in range(nTrans):
    for step in range(steps_per_T):
        y = rk4(dyn, t, y, dt) # estos seran iguales en los demas problemas 
        t += dt
    xi = y[2*n_orbits:3*n_orbits]
    eta = y[3*n_orbits:4*n_orbits] 
    tangent_form = np.sqrt(xi**2 + eta**2)
    y[2*n_orbits:3*n_orbits] = (xi / tangent_form)
    y[3*n_orbits:] = (eta / tangent_form)
    
# maximo esponente lyapunov

sum_log = np.zeros(n_orbits)
for period in range(nLyap):
    for step in range (steps_per_T):
        y = rk4(dyn, t, y, dt)
        t += dt
    xi = y[2*n_orbits:3*n_orbits]
    eta = y[3*n_orbits:]
    tangent_form = np.sqrt(xi**2 + eta**2)
    sum_log += log(tangent_form)
    y[2*n_orbits: 3*n_orbits] = (xi / tangent_form)
    y[3*n_orbits:4*n_orbits] = eta / tangent_form
    
# maximo exponente de lyapunov exponent por unidad de tiempo 
lambda_max =  sum_log / (nLyap*T)
imax = np.argmax (lambda_max)
gamma_at_max = gamma_values[imax]
lambda_at_max = lambda_max[imax]
print(f"Maximum point = ({gamma_at_max:.6f},{lambda_at_max:.6f})")
fig,ax= plt.subplots(figsize=(8,6))
ax.plot(gamma_values,lambda_max,linewidth=0.1,color='blue')
ax.axhline(y=0,linewidth=0.5,color='black')
ax.set_xlabel(r'$\gamma$', fontsize=16)
ax.set_ylabel(r'$\lambda_{\max}$', fontsize=16)
ax.tick_params(axis='both', labelsize=12)
ax.text(0.03, 0.97, rf'$\gamma = {gamma_at_max:.2f}$',
        transform=ax.transAxes,ha='left',va='top',fontsize=14)
ax.set_xlim(gamma_values[0],gamma_values[-1])
ax.set_box_aspect(0.65)
plt.tight_layout()


# guardar imagen 
plt.savefig('Lya.pdf', format='pdf', bbox_inches='tight', dpi = 800)
plt.show()



     
        
