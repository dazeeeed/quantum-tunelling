import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d #only for 3d plots
from calculations import calc_pm

def f(x,t):
    # return np.exp(-a*x*x)
    return np.exp(-a*(x-t)**2)

def NL(V):
    '''Nonlinear exponential, where V is potential.'''
    return np.exp(-1j*dt*V)

def DS(k):
    '''Dispersive exponential'''
    return np.exp(1j*dt*k**2 / (2*m))

a = 0.5
x_marg, step_x = 10, 0.01
V, m = 0, 1
dt = 0.1
x = np.arange(-x_marg, x_marg, step_x)
t_arr = np.arange(0, 2, dt)


#######################################################
# p_m COEFFICIENTS
p_m = calc_pm(m, len(x), step_x)

'''
try:
    n = int(input('How many derivatives do you want to calculate?\n'))
except:
    print("That was not a valid number.\n")
    exit()
'''
n = 2
#######################################################
# FOURIER TRANSFORM
psi = np.empty((n+1,len(x)), dtype=np.complex64)
#each row corresponds to (i+1)-th derivative, eg psi[0] = psi(x), psi[1] = dpsi(x)/dx

psi[0] = np.fft.fft(NL(V)*f(x,0))

for i in range(1,n+1):
    psi[i] = (1j*p_m)**i * psi[0] 
    psi[i] = np.fft.ifft(DS(p_m) * psi[i]) 
    # psi[i] = np.fft.ifft(psi[i]) 

psi[0] = np.fft.ifft(DS(p_m) * psi[0])

#######################################################
# PLOT CREATION
fig = plt.figure()
ax = plt.axes()

ax.plot(x,np.real(psi[0]), label='f(x)')
for i in range(1,n+1):
    ax.plot(x, np.real(psi[i]), label=f"$d^{i}/dx^{i} f(x)$", linewidth=5*np.random.rand(1))

# ANALYTIC DERIVATIVE FORMULAS
# ax.plot(x, -2*a*x*f(x), linestyle='-.', linewidth=3, label="Analytic f'(x)")
# ax.plot(x, 2*a*f(x)*(2*a*x*x-1), linestyle='-.', linewidth=3, label="Analytic f''(x)")

plt.legend()
plt.show()

#######################################################
# 3D Ploting

'''
fig = plt.figure()
ax = plt.axes(projection="3d")

ax.plot3D(x,f(x))
for i in range(n):
    ax.plot3D(x, np.real(psi[i]), np.imag(psi[i]), label="{} pochodna".format(i+1))

ax.set_xlabel('x')
ax.set_ylabel("Real")
ax.set_zlabel("Imag")

plt.show()
'''