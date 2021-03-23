import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d #only for 3d plots

def f(x):
    return np.exp(-a*x*x)
    # return np.exp(-a*x*x*x*x)

a = 1/2
x_marg= 10 #marginal value of x
step = 0.01
x = np.arange(-x_marg, x_marg, step)

#######################################################
# p_m COEFFICIENTS
n_x = len(x)
p_m = np.array([2*np.pi*m/(n_x*step) if m <= n_x/2 \
    else 2*np.pi*(m-n_x)/(n_x*step) \
    for m in range(n_x)])

''' ^^^ is equivalent with:
for m in range(n_x):
    if m <= n_x/2:
        p_m.append(2*np.pi*m/(n_x*step))
    else:
        p_m.append(2*np.pi*(m-n_x)/(n_x*step))
'''
try:
    n = int(input('How many derivatives do you want to calculate?\n'))
except:
    print("That was not a valid number.\n")
    exit()

#######################################################
# FOURIER TRANSFORM
fft_array = np.empty((n,n_x), dtype=np.complex64)
#each row corresponds to (i+1)-th derivative
fft_array[0] = (1j*p_m)**1 * np.fft.fft(f(x))

for i in range(1,n):
    fft_array[i] = (1j*p_m)**i * fft_array[0] 
    fft_array[i] = np.fft.ifft(fft_array[i]) 

fft_array[0] = np.fft.ifft(fft_array[0])

#######################################################
# PLOT CREATION
fig = plt.figure()
ax = plt.axes()

ax.plot(x,f(x), label='f(x)')
for i in range(n):
    ax.plot(x, np.real(fft_array[i]), label=f"$d^{i+1}/dx^{i+1} f(x)$", linewidth=5*np.random.rand(1))

ax.plot(x, -2*a*x*f(x), linestyle='-.', linewidth=3, label="Analytic f'(x)")
ax.plot(x, 2*a*f(x)*(2*a*x*x-1), linestyle='-.', linewidth=3, label="Analytic f''(x)")

plt.legend()
plt.show()

#######################################################
# 3D Ploting

'''
fig = plt.figure()
ax = plt.axes(projection="3d")

ax.plot3D(x,f(x))
for i in range(n):
    ax.plot3D(x, np.real(fft_array[i]), np.imag(fft_array[i]), label="{} pochodna".format(i+1))

ax.set_xlabel('x')
ax.set_ylabel("Real")
ax.set_zlabel("Imag")

plt.show()
'''