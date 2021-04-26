import numpy as np

def calc_pm(m, n_x, step_x):
    '''Calculation of p_m coefficients, where m-mass, n_x- lentgh of x, step_x- dx.
    p_m = 2*pi*f_m, where f_m=k (notation in articles)-frequencies of each 
    Fourier transform component.''' 
    
    return np.array([2*np.pi*m/(n_x*step_x) if m <= n_x/2 \
        else 2*np.pi*(m-n_x)/(n_x*step_x) \
        for m in range(n_x)])

''' ^^^ is equivalent with:
for m in range(n_x):
    if m <= n_x/2:
        p_m.append(2*np.pi*m/(n_x*step_x))
    else:
        p_m.append(2*np.pi*(m-n_x)/(n_x*step_x))
'''

def V_barrier(x, x_0, value=0, width=0):
    '''Potential barrier array creation. x_0 is a position of the barrier. 
    Value is a height of this barrier. Width is a width of the barrier.'''
    barrier = np.empty(len(x))
    for ix in range(len(x)):
        if x[ix] >= x_0 and x[ix] <= x_0 + width:
            barrier[ix] = value
        else:
            barrier[ix] = 0

    return barrier


'''
#######################################################
# 3D Ploting
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

'''
# ax.plot(x, np.abs(psi[0]), label='f(x)')
# for i in range(1,n+1):
#     ax.plot(x, np.abs(psi[i]), label=f"$d^{i}/dx^{i} f(x)$", linewidth=5*np.random.rand(1))
# ANALYTIC DERIVATIVE FORMULAS
# ax.plot(x, -2*a*x*f(x), linestyle='-.', linewidth=3, label="Analytic f'(x)")
# ax.plot(x, 2*a*f(x)*(2*a*x*x-1), linestyle='-.', linewidth=3, label="Analytic f''(x)")
'''