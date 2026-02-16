# Includes
import numpy as np
from scipy.optimize import minimize, differential_evolution

# Define orbital elements for each body

aEarth = 1.4765067 * 10**8
eEarth = 9.1669995 * 10**-3
iEarth = np.deg2rad(4.2422693 * 10**-3)
wEarth = np.deg2rad(6.64375167 * 10**1)
OEarth = np.deg2rad(1.4760836 * 10**1)

aApo = 1.3793939 * 10**8
eApo = 1.9097084 * 10**-1
iApo = np.deg2rad(3.3356539 * 10**0)
wApo = np.deg2rad(1.2919949 * 10**2)
OApo = np.deg2rad(2.0381969 * 10**2)

aYR = 3.7680703 * 10**8
eYR = 6.6164147 * 10**-1
iYR = np.deg2rad(3.4001497 * 10**0)
wYR = np.deg2rad(1.3429905 * 10**2)
OYR = np.deg2rad(2.7147904 * 10**2)

aATL = -3.9552667 * 10**7
eATL = 6.1469268 * 10**0
iATL = np.deg2rad(1.7512507 * 10**2)
wATL = np.deg2rad(1.2817255 * 10**2)
OATL = np.deg2rad(3.2228906 * 10**2)

Earth = (aEarth,eEarth,iEarth,OEarth,wEarth)
Apo = (aApo,eApo,iApo,OApo,wApo)
YR = (aYR,eYR,iYR,OYR,wYR)
ATL = (aATL,eATL,iATL,OATL,wATL)


# Define necessary functions

def kepler_to_r(a, e, i, Omega, omega, f): # Calculate radius vector  
    # Radius magnitude
    if e < 1:  # Ellipse case
        r = a * (1 - e**2) / (1 + e * np.cos(f))
    else:      # Hyperbola case
        r = a * (e**2 - 1) / (1 + e * np.cos(f))  # hyperbolic formula

    # Perifocal coordinates
    x_pf = r * np.cos(f)
    y_pf = r * np.sin(f)

    # Precompute trig
    cO = np.cos(Omega)
    sO = np.sin(Omega)
    ci = np.cos(i)
    si = np.sin(i)
    cw = np.cos(omega)
    sw = np.sin(omega)

    # Use ECI rotation matrix
    R11 =  cO*cw - sO*sw*ci
    R12 = -cO*sw - sO*cw*ci
    R13 =  sO*si

    R21 =  sO*cw + cO*sw*ci
    R22 = -sO*sw + cO*cw*ci
    R23 = -cO*si

    R31 =  sw*si
    R32 =  cw*si
    R33 =  ci

    # Apply transformation
    x = R11*x_pf + R12*y_pf
    y = R21*x_pf + R22*y_pf
    z = R31*x_pf + R32*y_pf

    return np.array([x, y, z])


def distance_squared(f_vals, orbit1, orbit2):
    f1, f2 = f_vals
    r1 = kepler_to_r(*orbit1, f1)
    r2 = kepler_to_r(*orbit2, f2)
    return np.linalg.norm(r1 - r2)**2


def generate_true_anomalies(e, num_points=500): # Function that finds true anomalies for both elyptical and hyperbolic orbits
    if e < 1:
        return np.linspace(0, 2*np.pi, num_points)
    else:
        f_max = np.arccos(-1 / e)
        return np.linspace(-f_max, f_max, num_points)

# Assign orbits (note that the lines below are changed for each orbital calculation)
orbit1 = Earth  # (a, e, i, Omega, omega)
orbit2 = ATL

# Generate true anomalies 
f_vals1 = generate_true_anomalies(orbit1[1], num_points=500)
f_vals2 = generate_true_anomalies(orbit2[1], num_points=500)

# Grid search
min_dist = 1e20
f1_min = 0
f2_min = 0

for f1 in f_vals1:
    for f2 in f_vals2:
        d2 = distance_squared([f1, f2], orbit1, orbit2)
        if d2 < min_dist:
            min_dist = d2
            f1_min = f1
            f2_min = f2

moid = np.sqrt(min_dist)
moid_au = moid / 1.495978707e8  # km -> AU

print("Grid MOID (AU):", moid_au)
print("True anomalies at MOID (radians):", f1_min, f2_min)

# Optimizer bounds adjusted for orbit type
def get_bounds(e1, e2):
    if e1 < 1:
        f1_bounds = (0, 2*np.pi)
    else:
        f_max1 = np.arccos(-1 / e1)
        f1_bounds = (-f_max1, f_max1)
    
    if e2 < 1:
        f2_bounds = (0, 2*np.pi)
    else:
        f_max2 = np.arccos(-1 / e2)
        f2_bounds = (-f_max2, f_max2)
    
    return [f1_bounds, f2_bounds]

bounds = get_bounds(orbit1[1], orbit2[1])

# Different optimization method
x0 = [f1_min, f2_min]  # from coarse grid

res = minimize(
    lambda x: distance_squared(x, orbit1, orbit2),
    x0,
    method='L-BFGS-B',
    bounds=bounds,
    options={
        'ftol': 1e-16,
        'gtol': 1e-12,
        'maxiter': 10000
    }
)

# Extract optimized true anomalies
f1_opt, f2_opt = res.x

moid = np.sqrt(res.fun) / 1.495978707e8

print("Refined MOID (AU):", moid)
print("True anomalies at refined MOID (radians):", f1_opt, f2_opt)
