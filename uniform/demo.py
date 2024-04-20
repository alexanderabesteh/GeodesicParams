from geodesicparams import *

from datetime import datetime
from os.path import dirname, abspath

pathway = str(dirname(dirname(abspath(__file__))))
workdir = pathway + "/uniform/"
date = (datetime.now()).strftime("%Y-%m-%d_%H-%M-%S")
config = [workdir, date]

import numpy as np
from mpmath import sin, cos, nstr, re, sqrt
import matplotlib.pyplot as plt

clear_directory(workdir + "temp/")
cos = np.vectorize(cos, "D")
sin = np.vectorize(sin, "D")

bh_mass = 1
energy = sqrt(0.95)  #1 #1.1 #1.2
rot = 0#0.8
ang_mom = 3#5 #2
light = 1
particle_light = 1
orbittype = "terminating"
initials = [0, 20, 0.85, 0.33]
dir_initials = [1, 1]
mCharge = 0
perm = 1
p_mass = 1
grav = 1
carter = 12
eCharge = 0 #2
cosmo = 0
nut = 0

out = 1 + sqrt(1 - rot**2)
inner = 1 - sqrt(1 - rot**2)

sol = solve_geodesic_orbit(bh_mass, rot, eCharge, cosmo, nut, mCharge, light, grav, perm, particle_light, energy, ang_mom, carter, p_mass, orbittype, config, initials, dir_initials)
mino = np.linspace(0, 2.7, 100)

#periods, g2, g3, int_init, inits = np.load(workdir + "temp/rdata_" + date + ".npy", allow_pickle = True)
#mino = np.linspace(0, 2 * float(periods[0]), 500)

#rList = sol[0](mino)
rList = []
thetaList = []
#thetaList = sol[1](mino)
phiList = sol[2](mino)

for i in mino:
    rList.append(sol[0](i))
    thetaList.append(sol[1](i))

y1 = rList * sin(phiList)
th1 = rList * cos(thetaList)
th2 = rList * sin(thetaList)
x1 = rList * cos(phiList)
x = rList * cos(phiList) * sin(thetaList)
y = rList * sin(phiList) * sin(thetaList)
z = rList * cos(thetaList)

for i in range(len(rList)):
    x1[i] = float(nstr(re(x1[i])))
    y1[i] = float(nstr(re(y1[i])))
    x[i] = float(nstr(re(x[i])))
    th1[i] = float(nstr(re(th1[i])))
    th2[i] = float(nstr(re(th2[i])))

    y[i] = float(nstr(re(y[i])))
    z[i] = float(nstr(re(z[i])))
    rList[i] = float(nstr(rList[i]))
    thetaList[i] = float(nstr(re(thetaList[i])))
    phiList[i] = float(nstr(re(phiList[i])))
#print(phiList)

figure, axe = plt.subplots()
#figure, axe = plt.subplots(subplot_kw = dict(projection = "3d"))
#axe.set_xlim([-20, 20])
#axe.set_ylim([-20, 20])
#axe.set_zlim([-20, 20])
axe.axhline(y=0, color='black')
axe.axvline(x=0, color='black')

#axe.plot(x, y, z)
#out_hor = plt.Circle((0, 0), out, color = "red", fill = False)
#in_hor = plt.Circle((0, 0), inner, color = "yellow", fill = False)
#axe.add_patch(out_hor)
#axe.add_patch(in_hor)
#axe.plot(th2, th1)
axe.plot(x1, y1)
#axe.plot(mino, phiList)
plt.show()
