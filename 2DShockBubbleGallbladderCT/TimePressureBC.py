import numpy as np
from scipy.interpolate import interp1d

pK = np.loadtxt('pressureBC.txt', delimiter=',', skiprows=0)

time = np.linspace(0.0,1E-06,40000)

p2 = interp1d(pK[:,0], pK[:,1], kind='linear')

def interpP(t):
    p = 0
    p = p2(t)
    return p
