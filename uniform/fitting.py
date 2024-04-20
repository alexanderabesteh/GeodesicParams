from geodesicparams import *

from datetime import datetime
from os.path import dirname, abspath
from astroquery.jplhorizons import Horizons
from astropy.coordinates import SkyCoord
from numpy import array
from mpmath import sqrt

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

pathway = str(dirname(dirname(abspath(__file__))))
workdir = pathway + "/uniform/"
date = (datetime.now()).strftime("%Y-%m-%d_%H-%M-%S")
config = [workdir, date, 15]

#init_theta = [1, 0.0001328, 0, 0.00005149, -19, 0.0001] 
init_theta = [1, 0.8, 0, 1, 12, sqrt(0.95), 3]
#init_theta = [1, 1, 0, 1, 1, 1, 1]
steps = 5000
print(fit_geodesic_orbit(data_points, init_theta, steps, config))
