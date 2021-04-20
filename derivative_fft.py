import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#from mpl_toolkits import mplot3d #only for 3d plots
from calculations import calc_pm, V_barrier

def f(x):
    return np.exp(-a*x*x)
    #return np.exp(-a*(x-t)**2)

def NL(V,dt=0.05):
    '''Nonlinear exponential, where V is potential.'''
    return np.exp(-1j*dt*V)

def DS(k, dt=0.05, m=1):
    '''Dispersive exponential'''
    return np.exp(1j*dt*k**2 / (2*m))

def eq1221(x,t, step_x=0.01, m=1, h=1):
    '''Equation 12.21 from some ultra-wise physics book.'''
    return (2*np.pi*(step_x**2+h**2*t**2/(4*m**2*step_x**2)))**(-1/2) * \
        np.exp(-x**2/(2*(step_x**2+h**2*t**2/(4*m**2*step_x**2))))

a = 0.5
h = 1
x_marg, step_x = 100, 0.01
m = 1
dt = 0.05
x = np.arange(-x_marg, x_marg, step_x)
V = V_barrier(x, 5, 0.5, 5)


#######################################################
# p_m COEFFICIENTS
p_m = calc_pm(m, len(x), step_x) #/(2*np.pi)

n = 2 # how many derivatives to count - NOT USED

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

y = eq1221(x,0.01)
#y = psi[0]
line, = ax.plot(0,0)
ax.set_xlim(-0.5*x_marg,0.5*x_marg)
ax.set_ylim(-0.01,0.6)
ax.plot(x, V)
time = ax.text(0.35, 0.9, "Time: 0 s",
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes,
        color='green', fontsize=15)

def animation_frame(i):
    global y, dt, p_m
    if not pause:
        time.set_text("Time: {} s".format(np.round(i*dt,2)))
        y = np.fft.fft(NL(V,dt) * y)

        #y = np.fft.ifft(DS(p_m,dt) * y)
        y = np.fft.ifft(DS(0.25*p_m - 60*dt, dt) * y)
        
        line.set_xdata(x)
        line.set_ydata(np.abs(y))
    return line,

pause = False
def onClick(event):
    global pause
    pause ^= True

fig.canvas.mpl_connect('button_press_event', onClick)

anim = animation.FuncAnimation(fig, func=animation_frame, frames=3*200, interval=10, repeat=False)
#uncomment below to save animation, needs imagemagick installed
#anim.save("/tmp/animation.gif", writer="imagemagick", fps=30)

#plt.legend()
plt.show()


