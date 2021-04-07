import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#from mpl_toolkits import mplot3d #only for 3d plots
from calculations import calc_pm

def f(x):
    return np.exp(-a*x*x)
    #return np.exp(-a*(x-t)**2)

def NL(V):
    '''Nonlinear exponential, where V is potential.'''
    return np.exp(-1j*dt*V)

def DS(k):
    '''Dispersive exponential'''
    return np.exp(1j*dt*k**2 / (2*m))

def eq1221(x,t, step_x=0.01, m=1, h=1):
    '''Equation 12.21 from some ultra-wise physics book.'''
    return (2*np.pi*(step_x**2+h**2*t**2/(4*m**2*step_x**2)))**(-1/2) * \
        np.exp(-x**2/(2*(step_x**2+h**2*t**2/(4*m**2*step_x**2))))

a = 0.5
h = 1
x_marg, step_x = 10, 0.01
V, m = 0, 1
dt = 0.05
x = np.arange(-x_marg, x_marg, step_x)

#######################################################
# p_m COEFFICIENTS
p_m = calc_pm(m, len(x), step_x) #/ (2*np.pi)

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
#each row corresponds to i-th derivative, eg psi[0] = psi(x), psi[1] = dpsi(x)/dx

for iterations in range(1):
    psi[0] = np.fft.fft(NL(V)*f(x))
    # for i in range(1,n+1):
        # #This loop is for calculation of derivatives
        # psi[i] = (1j*p_m)**i * psi[0] 
        # psi[i] = np.fft.ifft(DS(p_m) * psi[i]) 
    psi[0] =  np.fft.ifft(DS(p_m) *psi[0])

#######################################################
# PLOT CREATION
fig = plt.figure()
ax = plt.axes()

# ax.plot(x, np.abs(psi[0]), label='f(x)')
# for i in range(1,n+1):
#     ax.plot(x, np.abs(psi[i]), label=f"$d^{i}/dx^{i} f(x)$", linewidth=5*np.random.rand(1))

# ANALYTIC DERIVATIVE FORMULAS
# ax.plot(x, -2*a*x*f(x), linestyle='-.', linewidth=3, label="Analytic f'(x)")
# ax.plot(x, 2*a*f(x)*(2*a*x*x-1), linestyle='-.', linewidth=3, label="Analytic f''(x)")

# y = eq1221(x,dt)
y = psi[0]
line, = ax.plot(0,0)
ax.set_xlim(-x_marg,x_marg)
ax.set_ylim(-1,1)

def animation_frame(i):
    global y, dt
    if not pause:
        y = np.fft.fft(NL(V) * y)
        y = np.fft.ifft(DS(p_m) * y)
        line.set_xdata(x)
        line.set_ydata(np.abs(y))
        print('Time: {} s'.format(np.round(i*dt,2)))
    return line,

pause = False
def onClick(event):
    global pause
    pause ^= True
fig.canvas.mpl_connect('button_press_event', onClick)

anim = animation.FuncAnimation(fig, func=animation_frame, frames=201, interval=1, repeat=False)
#uncomment below to save animation, needs imagemagick installed
#anim.save("/tmp/animation.gif", writer="imagemagick", fps=30)

#plt.legend()
plt.show()
