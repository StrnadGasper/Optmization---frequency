# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 22:06:47 2020

@author: gaspe
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from pylab import *
from scipy import optimize
"""
prva točka naloge, mamo koordinate x,y z napako
treba je najdit parametre fitane funkcije
"""

frekvenca = np.array([10,	43.42,	57.98,	74.96,	101.74,	145.69,	186.04,	238.2,	279.7,	334.6,	389,	429.9,	485.1,	521.1,	558,	611.8,	660.7,	706,	764.4,	805.4,	842.5,	870.6,	927,	998,	1060.4,	1127.5])

napetost = np.array([-16.98,	-15.9,	-14.74,	-13.18,	-10.74,	-7.81,	-6.07,	-4.58,	-3.72,	-2.9,	-2.29,	-1.95,	-1.56,	-1.37,	-1.22,	-1.01,	-0.82,	-0.7,	-0.58,	-0.52,	-0.46,	-0.43,	-0.34,	-0.27,	-0.21,	-0.15])

napake = np.zeros(26)
err= napake + 0.1

xos= np.linspace(0,1250,1250)
def funkcija(x,a,b,c):
    return a*np.exp(-b*x)+c

plt.errorbar(frekvenca,napetost,yerr=err,linestyle ='')
#plt.plot(xos,funkcija(xos))
plt.xlabel("Frekvenca [Hz]")
plt.ylabel("Napetost [mV]")
plt.show()



parametri, kovarianca = optimize.curve_fit(funkcija, frekvenca, napetost, p0=[0,0,1], sigma=err)
print(parametri)
print(kovarianca)


plt.errorbar(frekvenca,napetost,yerr=err,ls='')
plt.plot(xos,funkcija(xos,-19.018,5.90498e-03,-3.19185e-01))
plt.xlabel("Frekvenca[Hz]")
plt.ylabel("Napetost[mV]")
plt.show()
"""
def chi(y0,a):
    model = (y0*(doza))/(doza+a)
    chisq = np.sum(((odziv - model)/napake)**2)
    return chisq
print("Vrednost chi2:")
print(chi(106.317,24.762))
"""