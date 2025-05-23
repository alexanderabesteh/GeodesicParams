from geodesicparams import *

from datetime import datetime
from os.path import dirname, abspath
from astroquery.jplhorizons import Horizons
from astropy.coordinates import SkyCoord
from numpy import array
from mpmath import sqrt
import matplotlib.pyplot as plt
obj = Horizons(id='199', location='500@10',
               epochs={'start':'1920-01-01', 'stop':'2019-12-31',
                       'step':'1y'})
vec = obj.vectors(refplane = "earth", aberrations = "astrometric")
t_init = vec["datetime_jd"][0] #* 86400
#print(t_init)
vecs_au = SkyCoord(x = vec['x'], y = vec['y'], z = vec['z'], unit = 'au', frame = "icrs", representation_type = 'cartesian')

x = vecs_au.x.value.tolist()
y = vecs_au.y.value.tolist()
z = vecs_au.z.value.tolist()

data_points = array(list(zip(x, y, z)))
res = fit_ellipse_3d(data_points, 500)[0]

x1 = res[:, 0]
y2 = res[:, 1]
z3 = res[:, 2]
figure, axe = plt.subplots(subplot_kw = dict(projection = "3d"))
#axe.set_xlim([-20, 20])
#axe.set_ylim([-20, 20])
#axe.set_zlim([-20, 20])

axe.plot(x1, y2, z3, color = "red")
axe.scatter(x, y, z)
axe.set_title('Neptune Orbit Fitting from 1920 - 2019 in Astronomical Units')
#out_hor = plt.Circle((0, 0), out, color = "red", fill = False)
#in_hor = plt.Circle((0, 0), inner, color = "yellow", fill = False)
#axe.add_patch(out_hor)
#axe.add_patch(in_hor)
#axe.plot(th2, th1)
#axe.plot(x1, y1)
#axe.plot(mino, phiList)
plt.show()

pathway = str(dirname(dirname(abspath(__file__))))
workdir = pathway + "/tests/"
date = (datetime.now()).strftime("%Y-%m-%d_%H-%M-%S")
config = [workdir, date, 15]

#init_theta = [1, 0.0001328, 0, 0.00005149, -19, 0.0001] 
init_theta = [1, 0.8, 0, 1, 12, sqrt(0.95), 3]
#init_theta = [1, 1, 0, 1, 1, 1, 1]
steps = 5000
#print(fit_geodesic_orbit(data_points, init_theta, steps, config))
