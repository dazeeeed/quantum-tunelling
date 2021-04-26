import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from barrier import Barrier


class Animation:
    def __init__(self, model):
        self.h = model["h"]
        self.m = model["m"]
        self.x = model["x"]
        self.dt = model["dt"]
        self.p_m = model["p_m"]
        self.barrier = model["barrier"]
        self.value = model["value"]
        self.x_min = model["x_lim"][0]
        self.x_max = model["x_lim"][1]


    def start(self):
        fig = plt.figure()
        ax = plt.axes()

        self.y = eq1221(self.x, 0.01, m=self.m, h=self.h)
        line, = ax.plot(0, 0)
        ax.set_xlim(self.x_min, self.x_max)
        ax.set_ylim(-0.01, 4)
        ax.plot(self.x, self.barrier)
        # time = ax.text(0.35, 0.9, "Time: 0 s",
        #                verticalalignment='bottom', horizontalalignment='right',
        #                transform=ax.transAxes,
        #                color='green', fontsize=15)

        def animation_frame(i):
            if not pause:
                # time.set_text("Time: {} s".format(np.round(i * self.dt, 2)))
                self.y = np.fft.fft(NL(self.barrier, self.dt) * self.y)

                self.y = np.fft.ifft(DS(0.25 * self.p_m - 60 * self.dt, self.dt) * self.y)

                line.set_xdata(self.x)
                line.set_ydata(np.abs(self.y))
            return line,

        pause = False

        def onClick(event):
            global pause
            pause ^= True

        fig.canvas.mpl_connect('button_press_event', onClick)

        anim = animation.FuncAnimation(fig, func=animation_frame, frames=3 * 200, interval=10, repeat=False)

        plt.show()


class Physics:
    def __init__(self, x_min, x_max, m):
        self.x_min = x_min
        self.x_max = x_max
        self.step_x = 0.01
        self.dt = 0.1
        self.h = 1 #6.626e-34
        self.m = m

        self.x = np.arange(self.x_min, self.x_max, self.step_x)
        self.p_m = calc_pm(len(self.x), self.step_x)


    def make_barrier(self, x_0, value, width, barrier_type):
        self.value = value
        self.barrier = Barrier(self.x, x_0, value, width, barrier_type).get_barrier()

    def get_model(self):
        return {"h": self.h,
                "m": self.m,
                "dt": self.dt,
                "barrier": self.barrier,
                "value": self.value,
                "x": self.x,
                "p_m": self.p_m,
                "x_lim": [self.x_min, self.x_max]}


def calc_pm(n_x, step_x):
    """Calculation of p_m coefficients where n_x- lentgh of x, step_x- dx.
    p_m = 2*pi*f_m, where f_m=k (notation in articles)-frequencies of each
    Fourier transform component."""

    return np.array([pm_1(m, n_x, step_x) if condition(m, n_x) else pm_2(m, n_x, step_x) for m in range(n_x)])


def condition(m, n_x):
    return m < n_x / 2


def pm_1(m, n_x, step_x):
    return 2 * np.pi * m / (n_x * step_x)


def pm_2(m, n_x, step_x):
    return 2 * np.pi * (m - n_x) / (n_x * step_x)


def NL(V,dt=0.05):
    """Nonlinear exponential, where V is potential."""
    return np.exp(-1j*dt*V)


def DS(k, dt=0.05, m=1):
    """Dispersive exponential"""
    return np.exp(1j*dt*k**2 / (2*m))


def eq1221(x,t=0.01, step_x=0.01, m=1, h=1):
    """Equation 12.21 from some ultra-wise physics book."""
    return (2*np.pi*(step_x**2+h**2*t**2/(4*m**2*step_x**2)))**(-1/2) * \
        np.exp(-x**2/(2*(step_x**2+h**2*t**2/(4*m**2*step_x**2))))
