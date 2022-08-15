#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  3 15:53:26 2021

@author: MadiStoddard
"""

import numpy as np
import scipy.stats as stats 
from matplotlib import pyplot as pp

#Transmissibility
fig, ax = pp.subplots()
start = -10
stop = 50
x = np.linspace (start, stop, 70*100) 
y1 = stats.gamma.pdf(x, a=20.52, scale=1/1.59, loc = -12.27)
y1cdf = stats.gamma.cdf(x, a=20.52, scale=1/1.59, loc = -12.27)

#scale = 1/rate
#Figure 1A
pp.plot(x, y1) 
pp.xlabel("Day POS")
pp.ylabel("Probability density (%)")
#pp.show()

#Symptom onset to death
#fisk.pdf(x, c, loc, scale)
y2 = stats.fisk.pdf(x, scale = 31.184, loc = -14.513, c = 6.802)


y2cdf = 1 - stats.fisk.cdf(x, scale = 31.184, loc = -14.513, c = 6.802)

pp.plot(x, y2) 
pp.xlabel("Day POS")
pp.legend(["Transmission","Death"])
#pp.ylabel("Death probability density (%)")
pp.savefig("C:/Users/amper/Desktop/work/Endemicity paper/Endemicity figures/Fig1A.tiff", dpi=600, facecolor='w', edgecolor='w',pad_inches=0.)
pp.show()

#Figure 1B
pp.plot(x, (1-y1cdf)*y2, color = 'g')
pp.xlabel("Day POS")
pp.ylabel("Loss of transmissibility (%)")
pp.savefig("C:/Users/amper/Desktop/work/Endemicity paper/Endemicity figures/Fig1B.tiff", dpi=600, facecolor='w', edgecolor='w',pad_inches=0.)
pp.show()

#Transmission | fatality
overlap = 0
for i in range(len(x)):
    overlap += y1[i]*y2cdf[i]
overlap = overlap*(stop-start)/len(x)
print(overlap)

#Total transmission 
integral = 0
for i in range(len(x)):
    integral += y1[i]
integral = integral*(stop-start)/len(x)
print(integral)

#Transmission loss
loss = integral - overlap
print(loss)

#transmission loss, alt.
loss2 = 0
for i in range(len(x)):
    loss2 += (1-y1cdf[i])*y2[i]
loss2 = loss2*(stop-start)/len(x)
print(loss2)