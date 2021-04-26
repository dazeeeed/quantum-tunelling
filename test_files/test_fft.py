import numpy as np
import matplotlib.pyplot as plt
from calculations import calc_pm

def f(a, x, x_shift=0, t=0):
    # return np.exp(-a*x*x)
    return np.exp(-a*(x-x_shift-t)**2)

def condition(x, x_0, width):
    return x >= x_0 and x <= x_0 + width

def V(x, x_0, value=0, width=0):
    '''x_0 is a position of the barrier. Value is a height of this barrier. Width is a width of the barrier'''
    barrier = np.empty(len(x))
    for ix in range(len(x)):
        if condition(ix, x_0, width):
            barrier[ix] = value
        else:
            barrier[ix] = 0

    # this returns a list, but we need np.array
    # barrier = [value if condition(ix, x_0, width) else 0 for ix in x]
    return barrier

def NL(dt, V):
    return np.exp(-1j * dt * V)

def DS(dt, k, m):
    return np.exp(-1j * dt * k**2 / (2*m))



def derivative(psi, p_m, n):
    '''psi is a complex numpy array, with size (length, ). Calculate n-th order derivative'''
    for i in range(1, n+1):
        psi_der = (1j*p_m)**i * psi
        psi_der = np.fft.ifft(psi_der)

    return psi_der

def init_psi(shape, function):
    psi = np.empty(shape, dtype=np.complex64)
    psi = np.fft.fft(function)

    return psi


# this last 2 functions was used only for FFT usage testing
def ft(psi_shape, a, x, p_m, x_shift=0):
    # FOURIER TRANSFORM
    psi = np.empty(psi_shape, dtype=np.complex64)
    #each row corresponds to (i+1)-th derivative, eg psi[0] = psi(x), psi[1] = dpsi(x)/dx

    psi[0] = np.fft.fft(f(a, x, x_shift=x_shift))

    for i in range(1,3):
        psi[i] = (1j*p_m)**i * psi[0] 
        psi[i] = np.fft.ifft(psi[i]) 

    psi[0] = np.fft.ifft(psi[0])

    return psi

def plot_results(x, psi, x_shift=0):
    fig, ax = plt.subplots(1,3, figsize=(15, 7))

    ax[0].plot(x,np.real(psi[0]), label='f(x)')
    ax[0].plot(x, V(x, 1.5, 2, 2), label='Barrier')
    ax[0].legend()

    for i in range(1, psi.shape[0]):
        ax[i].plot(x, np.real(psi[i]), label=f"$d^{i}/dx^{i} f(x)$", linewidth=2)
        ax[i].legend()

    # ANALYTIC DERIVATIVE FORMULAS
    ax[1].plot(x, -2*a*x*f(a, x), linestyle='-.', linewidth=3, label="Analytic f'(x)")
    ax[1].legend()
    ax[2].plot(x, -2*a*(1-2*a*x**2)*f(a, x), linestyle='-.', linewidth=3, label="Analytic f''(x)")
    ax[2].legend()

    plt.show()

if __name__ == '__main__':
    # simulation params
    a = 0.25
    x_marg, dx = 20, 0.01
    x_shift = -10
    m = 1
    x = np.arange(-x_marg, x_marg, dx)

    lx = len(x)
    p_m = calc_pm(m, lx, dx)

    # this psi matrix keeps f(x), 1st and 2nd order derivative. Only for testing FFT usage
    # psi = ft((3, lx),a, x, p_m, x_shift=x_shift)
    # plot_results(x, psi, x_shift=x_shift)

    # array with f(x) function
    psi = init_psi(lx, f(a, x))

    # set our barrier
    v = V(x, 1.5, 0, 0)

    # non-linear term
    nl = NL(0.1, v)

    psi = psi * nl

    # fourier transform psi_x -> psi_k
    psi = np.fft.fft(psi)

    # Dispersice term
    psi = psi * DS(0.1, 1, 1)

    #Inverse fourier transform
    psi = np.fft.ifft(psi)

    plt.plot(x, np.real(psi))
    plt.show()


    