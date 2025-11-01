#Throwing 10,000 pairs of 10 sided dice, summing each pair, counting the occurence of each outcome.
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(210055795)

n_samples = 10000
dice = rng.integers(1, 11, size=(n_samples, 2))
sums = np.sum(dice, axis=1)

counts = np.zeros(19)
for s in sums:
    counts[s-2] += 1
    
pdf = counts / float(n_samples)

sums = np.arange(2,21)
plt.bar(sums, pdf, align='center', alpha=0.5)
plt.xticks(sums)
plt.xlabel('Sum')
plt.ylabel('Probability')
plt.title('Monte Carlo Simulation of Sum of Two 10-Sided Dice')
plt.show()

total=np.sum(dice, axis=1)
mean = (np.mean(total))
sigma = (np.std(total))
print("SD", sigma)
print("Mean", mean)

#%% Question 1d

rng = np.random.default_rng(seed = 210055795)

arr1=rng.integers(low=1,high=11, size=(10000,10))

total=np.sum(arr1, axis=1)

plt.hist(total,bins=91)
plt.xticks([10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100])
plt.xlabel('Possible Outcomes from Sum of 10 Dice')
plt.ylabel('Number of Outcomes')
plt.title('Monte Carlo simulation of Sum of 10000 Throws of 10-Sided Dice')
print(np.mean(total))
print(np.std(total))
x=np.arange(1,101,1)
g=500*(np.exp(-((x-55.0565)**2)/(2*(9.088009009128458**2))))
plt.plot(x,g, color='green')
plt.legend(['Gaussian Distribution','Data'])

#%% Question 2a

data = np.loadtxt('FIRAS.txt')

x = data[:,0]
y = data[:,1]
error = data[:,3]

plt.plot(x,y, label='Spectral Radiance, B_v')
plt.xlabel('Frequency, f, (cm^-1)')
plt.ylabel('FIRAS monopole spectrum, B_v, (MJy/sr)')
plt.title('Spectral Radiance Data vs Frequency')
plt.errorbar(x, y, yerr=error, fmt=' ', label='Error')
plt.legend()
plt.grid()
plt.show()

print(np.mean(y))
print(np.std(y))

#%% Question 2b
data = np.loadtxt('FIRAS.txt')

T = 2.725
k_B = 1.380649*10**-23
h = 6.62607015*10**-34
c = 299792458
v = data[:,0]*c*100
residuals = (5,9,15,4,19,-30,-30,-10,32,4,-2,13,-22,8,8,-21,9,12,11,-29,-46,58,6,-6,6,-17,6,26,-12,-19,8,7,14,-33,6,26,-26,-6,8,26,57,-116,-432)

B_v = (2*v**2/c**2)*(h*v/np.exp(h*v/(k_B*T)-1))

plt.plot(B_v)
plt.xlabel('Frequecy, f, (Hz)')
plt.ylabel('Flux, \u03D5, (kJy/sr)')
plt.title('Theoretical Black Body Curve')
plt.show()

data = np.loadtxt('FIRAS.txt')

T = 2.725
k_B = 1.380649*10**-23
h = 6.62607015*10**-34
c = 299792458
v = data[:,0]*c*100
residuals = (5,9,15,4,19,-30,-30,-10,32,4,-2,13,-22,8,8,-21,9,12,11,-29,-46,58,6,-6,6,-17,6,26,-12,-19,8,7,14,-33,6,26,-26,-6,8,26,57,-116,-432)

B_v = (2*v**2/c**2)*(h*v/np.exp(h*v/(k_B*T)-1))

plt.plot(B_v)
plt.plot(residuals, 'o')
plt.xlabel('Frequecy, f, (Hz)')
plt.ylabel('Flux, \u03D5, (kJy/sr)')
plt.title('Theoretical Black Body Curve with Residuals')
plt.show()

data = np.loadtxt('FIRAS.txt')

T = 2.725
k_B = 1.380649*10**-23
h = 6.62607015*10**-34
c = 299792458
v = data[:,0]*c*100
residuals = (5,9,15,4,19,-30,-30,-10,32,4,-2,13,-22,8,8,-21,9,12,11,-29,-46,58,6,-6,6,-17,6,26,-12,-19,8,7,14,-33,6,26,-26,-6,8,26,57,-116,-432)


def black_body(v, T):
    return (2*v**2/c**2)*(h*v/(np.exp(h*v/k_B*T)-1))

sigma_T = 0.001
bb_curve = black_body(v, T)
bb_curve_upper = black_body(v, T + sigma_T)
bb_curve_lower = black_body(v, T - sigma_T)

fig, axs = plt.subplots(3, 1, sharex=True)

axs[0].plot(v, bb_curve, label='T = 2.725 K')
axs[1].plot(v, bb_curve_upper, label='T=2.726 K', color='red')
axs[2].plot(v, bb_curve_lower, label='T=2.724 K', color='green')

fig.text(0.06, 0.5,'Flux, \u03D5, (kJy/sr)', ha='center', va='center', rotation='vertical')
fig.suptitle('Theoretical Black Body Curve at T = 2.725 K \u00B1 1\u03C3')
axs[0].legend()
axs[1].legend()
axs[2].legend()

plt.xlabel('Frequecy, f, (Hz)')
plt.show()

#%% Question 2c

import numpy as np
from scipy.optimize import minimize

def planck(v, T):
    h = 6.62607004e-34
    c = 299792458
    k = 1.38064852e-23
    return 2 * h * v**3 / c**2 / (np.exp(h * v / k / T) - 1)

data = np.loadtxt('FIRAS.txt')
nu = data[:, 0] 
spec = data[: 1]
err = data[:, 3]
residual = data[: 2]

def chi2(T):
    return np.sum(residual ** 2)

result = minimize(chi2, 2.725)

print("Best-fitting temerature: %.6f K" % result.x[0])
print("Minimum chi-squared: %.6f" % result.fun)

cov = np.diag(err**2)

print("Covariance matrix:")
print(cov)

#%% Question 3a

import numpy as np
import sys
import os.path
from astropy.io import fits
import matplotlib
import matplotlib.pyplot as plt

hdulist = fits.open('GalCat-1.fits')
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

#splitting data
mass_halo = tbdata.field(0)
data_array = np.array([mass_halo, RA, Dec, z_spec, halo_id, dx, dy, dz, N_sat, mag]).T
flag_RA = data_array[:,1]<(5*60)

flag_mass_halo_central = data_array[:,0]>0
flag_mass_halo_sattelite = data_array[:,0]<0

data_array_sub_cent = data_array[flag_mass_halo_central,:]
data_array_sub_sat = data_array[flag_mass_halo_sattelite,:]



print('Central galaxy:', flag_mass_halo_cental)
print('Satellite galaxy:', flag_mass_halo_sattelite)

plt.hist(data_array_sub_cent[:,3], bins=500, label='Central Galaxy')
plt.hist(data_array_sub_sat[:,3], bins=500, label='Satellite Galaxy')
plt.title('Redshift Distribution for Central and Satellite Galaxies')
plt.xlabel('Redshift, z')
plt.ylabel('Number of Galaxies')
plt.legend()
plt.show()


#second part
hdulist = fits.open('GalCat-1.fits')
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
 

data_array = np.array([mass_halo, RA, Dec, z_spec, halo_id, dx, dy, dz, N_sat, mag]).T
 

flag_mass_halo_cental= data_array[:,0]>0
flag_mass_halo_sattelite = data_array[:,0]<0
data_array_sub_cent=data_array[flag_mass_halo_cental,:]
data_array_sub_sat=data_array[flag_mass_halo_sattelite,:]
print('Central galaxy:', flag_mass_halo_cental)
print('Satellite galaxy:', flag_mass_halo_sattelite)
ZCent=(data_array_sub_cent[:,3])
ZSat=(data_array_sub_sat[:,3])
plt.hist(ZCent,bins=500)
plt.legend(['Central Galaxy','Satellite Galaxy'])
plt.xlabel('Redshift,z')
plt.ylabel('Number of Galaxies')
plt.title('Redshift Distribution for Central and Satellite Galaxies')

ax = plt.gca()
p = ax.patches
ZCentH=np.array(([patch.get_height() for patch in p]))

plt.close()

plt.hist(ZSat,bins=500)
ax1=plt.gca()
m=ax1.patches
plt.close()
ZSatH=np.array(([patch.get_height() for patch in m]))
ZFrac=ZSatH/(ZSatH+ZCentH)
x=np.linspace(0,1.4,500)
plt.plot(x,ZFrac,'-', linewidth=0.6)
plt.xlabel('Redshift, z')
plt.ylabel('Satellite Fraction, f_sat(z)')
plt.title('Satellite Fraction as a Function of Redshift')

#%% Question 3b
hdulist = fits.open('GalCat-1.fits')
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



data_array = np.array([mass_halo, RA, Dec, z_spec, halo_id, dx, dy, dz, N_sat, mag]).T

 
flag_mass_halo_cental= data_array[:,0]>0
flag_mass_halo_sattelite = data_array[:,0]<0
data_array_sub_cent=data_array[flag_mass_halo_cental,:]
data_array_sub_sat=data_array[flag_mass_halo_sattelite,:]
print('Central galaxy:', flag_mass_halo_cental)
print('Sattelite galaxy:', flag_mass_halo_sattelite)
ZCent=(data_array_sub_cent[:,3])
ZSat=(data_array_sub_sat[:,3])
plt.hist(ZCent,bins=500)
plt.legend(['Central Galaxy','Sattelite Galaxy'])
plt.xlabel('Redshift,z')
plt.ylabel('Number of Galaxies')
plt.title('Redshift Distribution for Central and Sattelite Galaxies')

ax = plt.gca()
p = ax.patches
ZCentH=np.array(([patch.get_height() for patch in p]))
plt.close()
plt.hist(ZSat,bins=500)
ax1=plt.gca()
m=ax1.patches
plt.close()
ZSatH=np.array(([patch.get_height() for patch in m]))
ZFrac=ZSatH/(ZSatH+ZCentH)
x=np.linspace(0,1.4,500)
plt.plot(x,ZFrac,'-', linewidth=0.6)
plt.xlabel('Redshift, z')
plt.ylabel('Sattelite Fraction, f_sat(z)')
plt.title('Sattelite Fraction as a Function of Redshift')
plt.close()
centmass = np.log((data_array_sub_cent[:,0]))
plt.hist(centmass, bins=500)
plt.xlabel('Mass of Haloes, (Solar Masses)')
plt.ylabel('Number of Haloes')
plt.title('Number of Haloes as Function of their Mass')
#%% Question 3c
hdulist = fits.open('GalCat-1.fits')
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



data_array = np.array([mass_halo, RA, Dec, z_spec, halo_id, dx, dy, dz, N_sat, mag]).T

 
flag_mass_halo_cental= data_array[:,0]>0
flag_mass_halo_sattelite = data_array[:,0]<0
data_array_sub_cent=data_array[flag_mass_halo_cental,:]
data_array_sub_sat=data_array[flag_mass_halo_sattelite,:]
print('Central galaxy:', flag_mass_halo_cental)
print('Sattelite galaxy:', flag_mass_halo_sattelite)
ZCent=(data_array_sub_cent[:,3])
ZSat=(data_array_sub_sat[:,3])
plt.hist(ZCent,bins=500)
plt.legend(['Central Galaxy','Sattelite Galaxy'])
plt.xlabel('Redshift,z')
plt.ylabel('Number of Galaxies')
plt.title('Redshift Distribution for Central and Sattelite Galaxies')

ax = plt.gca()
p = ax.patches
ZCentH=np.array(([patch.get_height() for patch in p]))
plt.close()
plt.hist(ZSat,bins=500)
ax1=plt.gca()
m=ax1.patches
plt.close()
ZSatH=np.array(([patch.get_height() for patch in m]))
ZFrac=ZSatH/(ZSatH+ZCentH)
x=np.linspace(0,1.4,500)
plt.plot(x,ZFrac,'-', linewidth=0.6)
plt.xlabel('Redshift, z')
plt.ylabel('Sattelite Fraction, f_sat(z)')
plt.title('Sattelite Fraction as a Function of Redshift')
plt.close()
centmass = np.log((data_array_sub_cent[:,0]))
plt.hist(centmass, bins=500)
plt.xlabel('Mass of Haloes, (Solar Masses)')
plt.ylabel('Number of Haloes')
plt.title('Number of Haloes as Function of their Mass')

plt.close()

from scipy import stats
bin_edges = stats.mstats.mquantiles(ZCent, [0, 1/3, 2/3, 1])
n, bins, patches = plt.hist(ZCent, bins=bin_edges, log=True)
patches[0].set_fc('purple')
patches[1].set_fc('brown')
patches[2].set_fc('yellow')
plt.yticks([641152, 641153, 641154, 641155, 641156])
print(bin_edges)
plt.xlabel('Redshift, Z')
plt.ylabel('Number of Galaxies')
plt.title('Equi-Populated Histogram of Redshift Bands')

#%% Question 3d

hdulist = fits.open('GalCat-1.fits')
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



data_array = np.array([mass_halo, RA, Dec, z_spec, halo_id, dx, dy, dz, N_sat, mag]).T

 
flag_mass_halo_cental= data_array[:,0]>0
flag_mass_halo_sattelite = data_array[:,0]<0
data_array_sub_cent=data_array[flag_mass_halo_cental,:]
data_array_sub_sat=data_array[flag_mass_halo_sattelite,:]
print('Central galaxy:', flag_mass_halo_cental)
print('Satellite galaxy:', flag_mass_halo_sattelite)
ZCent=(data_array_sub_cent[:,3])
ZSat=(data_array_sub_sat[:,3])
plt.hist(ZCent,bins=500)
plt.legend(['Central Galaxy','Satellite Galaxy'])
plt.xlabel('Redshift,z')
plt.ylabel('Number of Galaxies')
plt.title('Redshift Distribution for Central and Satellite Galaxies')

ax = plt.gca()
p = ax.patches
ZCentH=np.array(([patch.get_height() for patch in p]))
plt.close()
plt.hist(ZSat,bins=500)
ax1=plt.gca()
m=ax1.patches
plt.close()
ZSatH=np.array(([patch.get_height() for patch in m]))
ZFrac=ZSatH/(ZSatH+ZCentH)
x=np.linspace(0,1.4,500)
plt.plot(x,ZFrac,'-', linewidth=0.6)
plt.xlabel('Redshift, z')
plt.ylabel('Satellite Fraction, f_sat(z)')
plt.title('Satellite Fraction as a Function of Redshift')
plt.close()
centmass = np.log((data_array_sub_cent[:,0]))
#plt.hist(centmass, bins=500)
plt.xlabel('Mass of Haloes, (Solar Masses)')
plt.ylabel('Number of Haloes')
plt.title('Number of Haloes as Function of their Mass')

plt.close()
Nsat = data_array[:,8]
halo_mass = data_array[:,0]
plt.scatter(mass_halo, Nsat, marker='o')
plt.xlabel('Halo Mass, (solar masses)')
plt.ylabel('Number of Satellite Galaxies, N_sat')
plt.title('Number of Satellites against Halo Mass')

#%% Question 4a

hdulist = fits.open('GalCat-1.fits')
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

data_array = np.array([mass_halo, RA, Dec, z_spec, halo_id, dx, dy, dz, N_sat, mag]).T

flag_mass_halo_cental= data_array[:,0]>0
flag_mass_halo_sattelite = data_array[:,0]<0
data_array_sub_cent=data_array[flag_mass_halo_cental,:]
data_array_sub_sat=data_array[flag_mass_halo_sattelite,:]
ZCent=(data_array_sub_cent[:,3])
ZSat=(data_array_sub_sat[:,3])

flag_halo_id = (data_array_sub_sat[:,4]==2)
data_array_sub_2=(data_array_sub_sat[flag_halo_id,:])
dx=(data_array_sub_2[:,5])
dr=abs(dx)
#print(dr)
n,bins,patches=plt.hist(dr, bins=50)
#print(flag_halo_id)
plt.xlabel('Distance to Center of Halo, dr, (Kpc/h)')
plt.ylabel('Number of Galaxies')
plt.title('Number of Galaxies as a Function of Halo Radius')

GalH=np.array(([patch.get_height() for patch in patches]))
x1 = np.linspace(0,4000,50)
#print(GalH)
fit=np.polyfit(x1,GalH,2)
y=((fit[0]*(x1**2))+(fit[1]*x1)+fit[2])
plt.plot(x1,y)
print(fit)

