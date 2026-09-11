from datetime import datetime
from os.path import abspath, dirname

from orbitronx import *

pathway = str(dirname(dirname(abspath(__file__))))
workdir = pathway + "/geodesic_orbit_demo/"
date = (datetime.now()).strftime("%Y-%m-%d_%H-%M-%S")
config = [workdir, date]

import matplotlib.pyplot as plt
import numpy as np
from mpmath import cos, sin, sqrt

clear_directory(workdir + "temp/")
cos = np.vectorize(cos, "D")
sin = np.vectorize(sin, "D")

bh_mass = 1
energy = 0.95 # 1 #1.1 #1.2
rot = 0.8  # 0.8
ang_mom = 3  # 5 #2
light = 1
particle_light = 1
orbittype = "bound"
initials = [0, 10, 0.85, 0.33]
dir_initials = [-1, 1]
mCharge = 0
perm = 1
p_mass = 1
grav = 1
carter = 12
eCharge = 0  # 2
cosmo = 0
nut = 0

out = 1 + sqrt(1 - rot**2)
inner = 1 - sqrt(1 - rot**2)

sol = solve_geodesic_orbit(
    bh_mass,
    rot,
    eCharge,
    cosmo,
    nut,
    mCharge,
    light,
    grav,
    perm,
    particle_light,
    energy,
    ang_mom,
    carter,
    orbittype,
    config,
    initials,
    dir_initials,
)
rdata = np.load(workdir + "temp/rdata_" + date + ".npy", allow_pickle=True)
periods, g2, g3, int_init, inits = rdata
mino_max = 6.0 * float(periods[0].real)

mino = np.linspace(0, mino_max, 500)


rList = sol[0](mino)
thetaList = sol[1](mino)
phiList = sol[2](mino)

rList = np.atleast_1d(np.real(np.array(rList, dtype=complex)).astype(float))
thetaList = np.atleast_1d(np.real(np.array(thetaList, dtype=complex)).astype(float))
phiList = np.atleast_1d(np.real(np.array(phiList, dtype=complex)).astype(float))

y1 = rList * sin(phiList)
th1 = rList * cos(thetaList)
th2 = rList * sin(thetaList)
x1 = rList * cos(phiList)
x = rList * cos(phiList) * sin(thetaList)
y = rList * sin(phiList) * sin(thetaList)
z = rList * cos(thetaList)

x1 = np.real(x1).astype(float)
y1 = np.real(y1).astype(float)
x = np.real(x).astype(float)
y = np.real(y).astype(float)
z = np.real(z).astype(float)
th1 = np.real(th1).astype(float)
th2 = np.real(th2).astype(float)

figure, axe = plt.subplots()
# figure, axe = plt.subplots(subplot_kw=dict(projection="3d"))
axe.set_xlim([-40, 40])
axe.set_ylim([-40, 40])
# axe.set_zlim([-20, 20])
axe.axhline(y=0, color="black")
axe.axvline(x=0, color="black")

# axe.plot(x, y, z)
out_hor = plt.Circle((0, 0), out, color = "red", fill = False)
in_hor = plt.Circle((0, 0), inner, color = "blue", fill = False)
axe.add_patch(out_hor)
axe.add_patch(in_hor)
axe.plot(th2, th1)
# axe.plot(x, y)
# axe.plot(mino, phiList)
plt.savefig(f"{workdir}/Images/figure_{date}.png")
plt.show()
