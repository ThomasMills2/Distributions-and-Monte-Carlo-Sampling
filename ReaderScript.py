import numpy as np
import sys
import os.path
from astropy.io import fits
import matplotlib 
import matplotlib.pyplot as plt

#open the file
hdulist = fits.open('./GalCat.fits')
tbdata = hdulist[1].data

mass_halo = tbdata.field(0) 
RA = tbdata.field(1)
Dec = tbdata.field(2)
z_spec = tbdata.field(5)
halo_id = tbdata.field(7)
dx = tbdata.field(8)
dy = tbdata.field(9)
dz = tbdata.field(10)
N_sat = tbdata.field(11)
mag = tbdata.field(13)

# Example Redshift histogram:
## set binwidth 
binwidth = 0.05
(n1, bins1, patches1) = plt.hist(z_spec, bins=np.arange(0., max(z_spec) + binwidth+0.1, binwidth), facecolor = 'blue')

# Example magnitude histogram
## set binwidth
binwidth = 0.5
(n1, bins1, patches1) = plt.hist(mag, bins=np.arange(15., max(mag) + binwidth+0.1, binwidth), facecolor = 'blue')
