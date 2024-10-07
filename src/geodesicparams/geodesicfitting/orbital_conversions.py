from mpmath import acos, tan, atan, sqrt, sin, pi

def true_anomaly(ecc, semi_maj):
    
    return lambda x : acos(((semi_maj * (1 - ecc**2) / x) - 1) / ecc)

def ecc_anomaly(ecc, semi_maj):

    nu = true_anomaly(ecc, semi_maj)
    denom = sqrt((1 + ecc) / (1 - ecc))

    return lambda x : 2 * atan(tan(nu(x) / 2) / denom)

def mean_anomaly(ecc, semi_maj):

    E = ecc_anomaly(ecc, semi_maj)

    return lambda x : E(x) - ecc * sin(E(x))

def mean_motion(ecc, semi_maj, mean0, t0):
    M = mean_anomaly(ecc, semi_maj)

    return lambda x, t1 : (M(x) - mean0) / (t1 - t0)

def orbital_elements(ecc, semi_maj, mean1, mean0, t1, t0):
    n = mean_motion(ecc, semi_maj, mean0, t0)(mean1, t1)
    standard_grav = semi_maj**3 * n**2
    orbital_period = 2 * pi * sqrt(semi_maj**3 / standard_grav)

    return n, standard_grav, orbital_period

def convert_newtonian(ecc, semi_maj, orbit_inc):

    return 0
