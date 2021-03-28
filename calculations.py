import numpy as np

def calc_pm(m, n_x, step_x):
    '''Calculation of p_m coefficients, where m-mass, n_x- lentgh of x, step_x- dx''' 
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