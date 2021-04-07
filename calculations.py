import numpy as np

def calc_pm(m, n_x, step_x):
    '''
    Calculation of p_m coefficients, where m-mass, n_x- lentgh of x, step_x- dx.
    p_m = 2*pi*f_m, where f_m=k (notation in articles)-frequencies of each 
    Fourier transform component.
    ''' 
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