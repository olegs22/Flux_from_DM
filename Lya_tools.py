import numpy as np
import math

def E(omega_m, omega_de, redshift):
    arg = omega_m*(1.0 + redshift)**3 + omega_de
    return np.sqrt(arg)

def tau(d, redshift):
    omega_m = 0.308
    omega_dm = 0.692
    omega_b = 0.0484

    Gamma = 1 #s^-1
    T = 2.0 #1.0K
    h = 0.678 #
    beta = 1.6

    evol = E(omega_m, omega_dm, redshift)

    tau = (1.41 * ((1.0 + redshift)**6) * ((omega_b*h**2)**2) * (d**beta)) / (T * h * evol * Gamma)

    return tau
