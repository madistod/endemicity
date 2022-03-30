#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 14:33:37 2020

@author: MadiStoddard
"""

import pylab as pp
import matplotlib as mp
import numpy as np
import scipy.integrate as integrate
import scipy.optimize as optimize
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatter 
from matplotlib.ticker import LogFormatterMathtext
import copy

residuals=[]
weights = []
class Parameterize_ODE():
    def __init__(self, S0, E0, I0, R0):
        self.S0 = S0
        self.E0 = E0
        self.I0 = I0
        self.R0 = R0
        
    def odes(self, t, y, p):
        Sv = y[0] #Susceptible vaccinated
        Su = y[1] #Susceptible unvaccinated
        Ev = y[2] #Exposed vaccinated
        Eu = y[3] #Exposed unvaccinated
        Iv = y[4] #Infectious vaccinated
        Iu = y[5] #Infectious unvaccinated
        Rv = y[6] #Recovered vaccinated
        Ru = y[7] #Recovered unvaccinated
        v_cum = y[8] #Cumulative cases among vaccinated
        u_cum = y[9] #Cumulative cases among unvaccinated
        deaths = y[10]
        
        A = p[0]
        R = p[1]
        C = p[2]
        D = p[3]*365/12 #scaled months -> days
        B = C/R #days; contact period
        bvac = 1.0 - p[4]
        c = 1.0 - p[5]
        sigma = p[6]/C
        mu = p[7]
        lam = p[8]
        f = p[9]
        case_mult = 1 - p[10] 
        
        dSv = -1/B*bvac*Sv*(Iu + Iv*c) + 1/D*Rv - Sv*mu + lam*f
        dSu = -1/B*Su*(Iu + Iv*c) + 1/D*Ru - Su*mu + lam*(1-f)
        
        dEv = 1/B*bvac*Sv*(Iu + Iv*c) - 1/A*Ev - Ev*mu
        dEu = 1/B*Su*(Iu + Iv*c) - 1/A*Eu - Eu*mu
        
        dIv = 1/A*Ev - 1/C*Iv - Iv*mu #- sigma*Iv*case_mult
        dIu = 1/A*Eu - 1/C*Iu - Iu*mu #- sigma*Iu
        
        dv_cum = 1/A*Ev
        du_cum = 1/A*Eu
        dDeaths = sigma*Iv*case_mult + sigma*Iu
        
        dRv = 1/C*Iv - 1/D*Rv - Rv*mu
        dRu = 1/C*Iu - 1/D*Ru - Ru*mu
        
        return [dSv, dSu, dEv, dEu, dIv, dIu, dRv, dRu, dv_cum, du_cum, dDeaths]
    
           
    def model(self, t, p):
        #y = [Sv, Su, Ev, Eu, Iv, Iu, R]
        S0 = self.S0
        E0 = self.E0
        I0 = self.I0
        R0 = self.R0
        f = p[9]
        y0 = [S0*f, S0*(1-f), E0*f, E0*(1-f), I0*f, I0*(1-f), R0*f, R0*(1-f), 0, 0, 0]
        r = integrate.solve_ivp(lambda t, y: self.odes(t, y, p), (t[0], t[-1]), y0, t_eval = t)
        y = np.array(r.y)
        return y 
 
    

#Initial conditions
I0 = 1.0/500. #initial infected population (fractional)
E0 = I0*2.56/10. #initial exposed population (fractional)
R0 = 0.08 #Initial recovered population (fractional)
S0 = 1.0 - I0 - E0 - R0 #Initial susceptible population (fractional)

#Disease parameters
A = 3. #days; latent period
C = 10. #days; infectious period
D = 18. #months; duration of immunity (range 3 mo - 2 yrs)
sigma = 0.007 #infection fatality rate

#Contact parameters
R = 5. #Individuals; Intrinsic reproductive number
B = C/R #days; contact period

#Vaccine parameters
f = 0.7  #fraction vaccinated
bvac = 0. # 1 - Relative rate of infection | vaccinated (fractional)
c = 0. # 1 - Relative rate of transmission | infected vaccinated (fractional)
case_mult = 0.7  # 1 - Relative risk of symptoms/mortality | infected vaccinated (fractional)

#Population dynamics parameters
lam = 0.009/365 #death rate
mu = lam #0.01/365 #birth rate

po = Parameterize_ODE(S0, E0, I0, R0)

params_optim = [A, R, C, D, bvac, c, sigma, mu, lam, f, case_mult] 

# Create a linearly spaced vector to run the optimized model at
days = 365*25
xt = np.linspace(0, days, days)

year24 = po.model(xt, params_optim)[10,days-365]
year25 = po.model(xt, params_optim)[10,-1]
yearly = (year25 - year24)*330 #yearly deaths
print("yearly deaths"+str(yearly))

output = po.model(xt, params_optim)
N = output[0,:]+output[1,:]+output[2,:]+output[3,:]+output[4,:]+output[5,:]+output[6,:]+output[7,:]

pp.plot(xt/365., N)
pp.xlabel("Time (years)")
pp.ylabel("Population (fractional)")
pp.show()

def multisensitivity(paramx, minx, maxx, paramy, miny, maxy, p, deaths = True):
    # Create a linearly spaced vector to run the optimized model at
    days = 365*10
    xt = np.linspace(0, days, days)
    
    points = 25
    if deaths == True:
        #Log scale for IFR
        prangey = np.logspace(np.log10(miny), np.log10(maxy), num=points)
    else:
        prangey = np.linspace(miny, maxy, num=points)
    print(prangey)
    prangex = np.linspace(minx, maxx, num=points)

    totarray = np.zeros((points,points))
    allarray = np.zeros((points,points))
    deathsarray = np.zeros((points,points))
    i2 = 0
    for valx in prangex:
        p[paramx] = valx
        print(i2)
        i1 = 0
        for valy in prangey:
            p[paramy] = valy
            
            #run model
            po = Parameterize_ODE(S0, E0, I0, R0)
            model_results = po.model(xt, p)
            
            #outcome metric
            population = 330
           
            allyear24 = model_results[8,days-365] + model_results[9,days-365]
            allyear25 = model_results[8,-1] + model_results[9,-1]
            ayearly = (allyear25 - allyear24)*population
        
            deathsyear24 = model_results[10,days-365]
            deathsyear25 = model_results[10,-1]
            deathsyearly = (deathsyear25 - deathsyear24)*population*1000

            allarray[i1,i2] = ayearly
            deathsarray[i1,i2] = deathsyearly
            i1+=1
        i2 += 1
        
    print("vmax3 ="+ str(np.amax(deathsarray)))
    
    #All cases
    print("Total cases"+str(np.amax(allarray)))
    
    params_list = ["Latency period (days)", "Reproductive number (individuals)", "Infectious period (days)", "Duration of Natural Immunity (months)", "Vaccine reduction in risk of infection", "Vaccine reduction in risk of transmission", "Infection Fatality Rate (%)", "Natural Death Rate (daily)", "Natural Birth Rate (daily)", "Fraction Compliant", "Vaccine reduction in risk of mortality"]
    
    if deaths == True:
        #Plotting yearly death toll heatmaps
        cmap1 = copy.copy(mp.cm.get_cmap("plasma"))
        cmap1.set_over(color = [0.96, 1., 0.132])
        fig, ax = pp.subplots()
        pp.title("R0 = "+str(p[1])+", Reduction in risk of infection = "+str(p[4]) )
        
        if np.amax(deathsarray) > 90:
            pp.pcolor(prangex, prangey, deathsarray, cmap = cmap1, vmin = 0, vmax = 650, shading = 'auto')
            cb = pp.colorbar() 
            cb.set_label(label="Yearly US deaths (thousands)", fontsize = 12)
            pp.yscale('log')
            levels = [25, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25400]
            CS = ax.contour(prangex, prangey, deathsarray, levels = levels, colors = 'darkturquoise')
            ax.clabel(CS, levels, fmt = '%.0f', fontsize = 12)
        elif np.amax(deathsarray) > 1:
            pp.pcolor(prangex, prangey, deathsarray, cmap = cmap1, vmin = 0, vmax = 650, shading = 'auto')
            cb = pp.colorbar() 
            cb.set_label(label="Yearly US deaths (thousands)", fontsize = 12)
            pp.yscale('log')
            levels = [1, 2, 4, 8, 16, 32, 64]
            CS = ax.contour(prangex, prangey, deathsarray, levels = levels, colors = 'darkturquoise')
            ax.clabel(CS, levels, fmt = '%.0f', fontsize = 12)
        else:
            pp.pcolor(prangex, prangey, deathsarray, cmap = cmap1, vmin = 0, vmax = 650, shading = 'auto')
            cb = pp.colorbar() 
            cb.set_label(label="Yearly US deaths (thousands)", fontsize = 12)
            pp.yscale('log')
            ax.text((maxx-minx)/2.+minx, (maxy-miny)/2.+miny, '-0-', fontsize = 12, color = 'darkturquoise')
        #pp.yscale('log')
        pp.yticks([0.05, 0.005, 0.0005], ["5", "0.5", "0.05"], fontsize = 12)
        pp.locator_params(axis='x', nbins=5)
        pp.scatter(18, 0.007, color = 'k')
    
    else:
        #Plotting yearly infections heatmaps
        cmap1 = copy.copy(mp.cm.get_cmap("Reds"))
        cmap1.set_under(color = 'limegreen')
        fig, ax = pp.subplots()
        pp.title("Fraction Vaccinated = "+str(p[9])+", Reduction in risk of infection = "+str(p[4]))
        
        if np.amax(deathsarray) > 90:
            pp.pcolor(prangex, prangey, allarray, cmap = cmap1, vmin = 0.01, vmax = 1100, shading = 'auto')
            cb = pp.colorbar() 
            cb.set_label(label="Yearly US infections (millions)", fontsize = 12)
            levels = [12, 25, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25400]
            CS = ax.contour(prangex, prangey, allarray, levels = levels, colors = 'k')
            ax.clabel(CS, levels, fmt = '%.0f', fontsize = 12)
        elif np.amax(deathsarray) > 1:
            pp.pcolor(prangex, prangey, allarray, cmap = cmap1, vmin = 0.01, vmax = 1100, shading = 'auto')
            cb = pp.colorbar() 
            cb.set_label(label="Yearly US infections (millions)", fontsize = 12)
            CS = ax.contour(prangex, prangey, allarray, colors = 'k')
            ax.clabel(CS, CS.levels, fmt = '%.0f', fontsize = 12)
        else:
            pp.pcolor(prangex, prangey, allarray, cmap = cmap1, vmin = 0.01, vmax = 1100, shading = 'auto')
            cb = pp.colorbar() 
            cb.set_label(label="Yearly US infections (millions)", fontsize = 12)
            ax.text((maxx-minx)/2.+minx, (maxy-miny)/2.+miny, '-0-', fontsize = 12, color = 'k')
    pp.xticks([3, 6, 9, 12, 15, 18, 21, 24], [3, 6, 9, 12, 15, 18, 21, 24], fontsize = 12)
    pp.ylabel(params_list[paramy], fontsize = 12)
    pp.xlabel(params_list[paramx], fontsize = 12)
    pp.tight_layout()

def linsensitivity(paramx, minx, maxx, p, color = 'k', linestyle = 'solid'):
    # Create a linearly spaced vector to run the optimized model at
    days = 365*25
    xt = np.linspace(0, days, days)
    print(days, len(xt))
    
    points = 50
    prangex = np.linspace(minx, maxx, num=points)

    allarray = np.zeros((points))
    deathsarray = np.zeros((points))
    i2 = 0
    for valx in prangex:
        p[paramx] = valx
        #print(i2)
        #run model
        po = Parameterize_ODE(S0, E0, I0, R0)
        model_results = po.model(xt, p)
        
        #outcome metric
        population = 330
       
        allyear24 = model_results[8,days-365] + model_results[9,days-365]
        allyear25 = model_results[8,-1] + model_results[9,-1]
        ayearly = (allyear25 - allyear24)*population
        
        deathsyear24 = model_results[10,days-365]
        deathsyear25 = model_results[10,-1]
        deathsyearly = (deathsyear25 - deathsyear24)*population

        allarray[i2] = ayearly
        deathsarray[i2] = deathsyearly
        i2 += 1
        
    print("vmax3 ="+ str(np.amax(deathsarray)))

    params_list = ["Latency period (days)", "Reproductive number (individuals)", "Infectious period (days)", "Duration of Natural Immunity (months)", "Vaccine reduction in risk of infection", "Vaccine reduction in risk of transmission", "Infection Fatality Rate", "Natural Death Rate (daily)", "Natural Birth Rate (daily)", "Fraction Compliant", "Vaccine reduction in IFR"]
    #pp.title("Fraction Vaccinated = "+str(p[9])+", Efficacy against mortality "+str(1-(1-p5[10])*(1-p5[4])))
    
    pp.plot(prangex, deathsarray, color = color, linestyle = linestyle)
    pp.xlabel(params_list[paramx], fontsize = 12)
    pp.ylabel("Yearly US Deaths (millions)")
    pp.tight_layout()

"""
#Figure 2: Infections vs R0, immunity
p5 = params_optim.copy()

p5[9] = 0.7 #70% compliance
p5[4] = 0. #no reduction in risk of infection
multisensitivity(3, 3, 24, 1, 2, 9, p5, deaths = False)
#pp.savefig('/Users/MadiStoddard/Desktop/work/Endemicity paper/Updated figures/Fig1D.tiff', dpi=600, facecolor='w', edgecolor='w',
#        pad_inches=0.)
pp.show()

p5[9] = 0.7 #70% compliance
p5[4] = 0.5 #50% reduction in risk of infection
multisensitivity(3, 3, 24, 1, 2, 9, p5, deaths = False)
#pp.savefig('/Users/MadiStoddard/Desktop/work/Endemicity paper/Updated figures/Fig1D.tiff', dpi=600, facecolor='w', edgecolor='w',
#        pad_inches=0.)
pp.show()

p5[9] = 0.7 #70% compliance
p5[4] = 0.9
multisensitivity(3, 3, 24, 1, 2, 9, p5, deaths = False)
#pp.savefig('/Users/MadiStoddard/Desktop/work/Endemicity paper/Updated figures/Fig1E.tiff', dpi=600, facecolor='w', edgecolor='w',
#        pad_inches=0.)
pp.show()

p5[9] = 1. #compliance
p5[4] = 0. #reduction in risk of infection
multisensitivity(3, 3, 24, 1, 2, 9, p5, deaths = False)
#pp.savefig('/Users/MadiStoddard/Desktop/work/Endemicity paper/Updated figures/Fig1F.tiff', dpi=600, facecolor='w', edgecolor='w',
#        pad_inches=0.)
pp.show()

p5[9] = 1. #70% compliance
p5[4] = 0.5 #90% reduction in risk of infection
multisensitivity(3, 3, 24, 1, 2, 9, p5, deaths = False)
#pp.savefig('/Users/MadiStoddard/Desktop/work/Endemicity paper/Updated figures/Fig1F.tiff', dpi=600, facecolor='w', edgecolor='w',
#        pad_inches=0.)
pp.show()

p5[9] = 1. #70% compliance
p5[4] = 0.9
multisensitivity(3, 3, 24, 1, 2, 9, p5, deaths = False)
#pp.savefig('/Users/MadiStoddard/Desktop/work/Endemicity paper/Updated figures/Fig1A.tiff', dpi=600, facecolor='w', edgecolor='w',
#        pad_inches=0.)
pp.show()

#Figure 3: Deaths as a function of IFR and immunity, 70% compliance
p5 = params_optim.copy()

p5[9] = 0.7 #70% compliance
p5[4] = 0. #0% reduction in risk of infection
p5[1] = 2  #R0
multisensitivity(3, 3, 24, 6, 0.0005, 0.05, p5)
#pp.savefig('/Users/MadiStoddard/Desktop/work/Endemicity paper/Updated figures/Fig1D.tiff', dpi=600, facecolor='w', edgecolor='w',
#        pad_inches=0.)
pp.show()

p5[9] = 0.7 #70% compliance
p5[4] = 0.
p5[1] = 5
multisensitivity(3, 3, 24, 6, 0.0005, 0.05, p5)
#pp.savefig('/Users/MadiStoddard/Desktop/work/Endemicity paper/Updated figures/Fig1E.tiff', dpi=600, facecolor='w', edgecolor='w',
#        pad_inches=0.)
pp.show()

p5[9] = 0.7 #70% compliance
p5[4] = 0. #90% reduction in risk of infection
p5[1] = 9
multisensitivity(3, 3, 24, 6, 0.0005, 0.05, p5)
#pp.savefig('/Users/MadiStoddard/Desktop/work/Endemicity paper/Updated figures/Fig1F.tiff', dpi=600, facecolor='w', edgecolor='w',
#        pad_inches=0.)
pp.show()

p5[9] = 0.7 #70% compliance
p5[4] = 0.5 #50% reduction in risk of infection
p5[1] = 2  #R0
multisensitivity(3, 3, 24, 6, 0.0005, 0.05, p5)
#pp.savefig('/Users/MadiStoddard/Desktop/work/Endemicity paper/Updated figures/Fig1D.tiff', dpi=600, facecolor='w', edgecolor='w',
#        pad_inches=0.)
pp.show()

p5[9] = 0.7 #70% compliance
p5[4] = 0.5
p5[1] = 5
multisensitivity(3, 3, 24, 6, 0.0005, 0.05, p5)
#pp.savefig('/Users/MadiStoddard/Desktop/work/Endemicity paper/Updated figures/Fig1E.tiff', dpi=600, facecolor='w', edgecolor='w',
#        pad_inches=0.)
pp.show()

p5[9] = 0.7 #70% compliance
p5[4] = 0.5 #90% reduction in risk of infection
p5[1] = 9
multisensitivity(3, 3, 24, 6,0.0005, 0.05, p5)
#pp.savefig('/Users/MadiStoddard/Desktop/work/Endemicity paper/Updated figures/Fig1F.tiff', dpi=600, facecolor='w', edgecolor='w',
#        pad_inches=0.)
pp.show()

p5[9] = 0.7 #70% compliance
p5[4] = 0.9
p5[1] = 2
multisensitivity(3, 3, 24, 6, 0.0005, 0.05, p5)
#pp.savefig('/Users/MadiStoddard/Desktop/work/Endemicity paper/Updated figures/Fig1A.tiff', dpi=600, facecolor='w', edgecolor='w',
#        pad_inches=0.)
pp.show()

p5[9] = 0.7 #70% compliance
p5[4] = 0.9 #90% reduction in risk of infection
p5[1] = 5
multisensitivity(3, 3, 24, 6, 0.0005, 0.05, p5)
#pp.savefig('/Users/MadiStoddard/Desktop/work/Endemicity paper/Updated figures/Fig1B.tiff', dpi=600, facecolor='w', edgecolor='w',
#        pad_inches=0.)
pp.show()

p5[9] = 0.7 #70% compliance
p5[4] = 0.9
p5[1] = 9
multisensitivity(3, 3, 24, 6, 0.0005, 0.05, p5)
#pp.savefig('/Users/MadiStoddard/Desktop/work/Endemicity paper/Updated figures/Fig1C.tiff', dpi=600, facecolor='w', edgecolor='w',
#        pad_inches=0.)
pp.show()


"""
#Figure 4: Deaths as a function of IFR and immunity, 100% compliance
p5 = params_optim.copy()
"""
p5[9] = 1. #70% compliance
p5[4] = 0. #90% reduction in risk of infection
p5[1] = 2
multisensitivity(3, 3, 24, 6, 0.0005, 0.05, p5)
#pp.savefig('/Users/MadiStoddard/Desktop/work/Endemicity paper/Updated figures/Fig2D.tiff', dpi=600, facecolor='w', edgecolor='w',
#        pad_inches=0.)
pp.show()
"""
p5[9] = 1. #70% compliance
p5[4] = 0.
p5[1] = 5
multisensitivity(3, 3, 24, 6, 0.0005, 0.05, p5)
#pp.savefig('/Users/MadiStoddard/Desktop/work/Endemicity paper/Updated figures/Fig2E.tiff', dpi=600, facecolor='w', edgecolor='w',
#        pad_inches=0.)
pp.show()
"""
p5[9] = 1. #70% compliance
p5[4] = 0. #90% reduction in risk of infection
p5[1] = 9
multisensitivity(3, 3, 24, 6, 0.0005, 0.05, p5)
#pp.savefig('/Users/MadiStoddard/Desktop/work/Endemicity paper/Updated figures/Fig2F.tiff', dpi=600, facecolor='w', edgecolor='w',
#        pad_inches=0.)
pp.show()
"""

p5[9] = 1. #70% compliance
p5[4] = 0.5 #90% reduction in risk of infection
p5[1] = 2
multisensitivity(3, 3, 24, 6, 0.0005, 0.05, p5)
#pp.savefig('/Users/MadiStoddard/Desktop/work/Endemicity paper/Updated figures/Fig2D.tiff', dpi=600, facecolor='w', edgecolor='w',
#        pad_inches=0.)
pp.show()
"""
p5[9] = 1. #70% compliance
p5[4] = 0.5
p5[1] = 5
multisensitivity(3, 3, 24, 6, 0.0005, 0.05, p5)
#pp.savefig('/Users/MadiStoddard/Desktop/work/Endemicity paper/Updated figures/Fig2E.tiff', dpi=600, facecolor='w', edgecolor='w',
#        pad_inches=0.)
pp.show()

p5[9] = 1. #70% compliance
p5[4] = 0.5 #90% reduction in risk of infection
p5[1] = 9
multisensitivity(3, 3, 24, 6, 0.0005, 0.05, p5)
#pp.savefig('/Users/MadiStoddard/Desktop/work/Endemicity paper/Updated figures/Fig2F.tiff', dpi=600, facecolor='w', edgecolor='w',
#        pad_inches=0.)
pp.show()

p5[9] = 1. #70% compliance
p5[4] = 0.9
p5[1] = 2
multisensitivity(3, 3, 24, 6, 0.0005, 0.05, p5)
#pp.savefig('/Users/MadiStoddard/Desktop/work/Endemicity paper/Updated figures/Fig2A.tiff', dpi=600, facecolor='w', edgecolor='w',
#        pad_inches=0.)
pp.show()
"""
p5[9] = 1. #70% compliance
p5[4] = 0.9 #90% reduction in risk of infection
p5[1] = 5
multisensitivity(3, 3, 24, 6, 0.0005, 0.05, p5)
#pp.savefig('/Users/MadiStoddard/Desktop/work/Endemicity paper/Updated figures/Fig2B.tiff', dpi=600, facecolor='w', edgecolor='w',
#        pad_inches=0.)
pp.show()
"""

p5[9] = 1. #70% compliance
p5[4] = 0.9
p5[1] = 9
multisensitivity(3, 3, 24, 6, 0.0005, 0.05, p5)
#pp.savefig('/Users/MadiStoddard/Desktop/work/Endemicity paper/Updated figures/Fig2C.tiff', dpi=600, facecolor='w', edgecolor='w',
 #       pad_inches=0.)
pp.show()


#Figure 5: Deaths vs. R0
p5 = params_optim.copy()
paramx = 1
minx = 1
maxx = 14
p5[9] = 0.7 #70% compliance
p5[4] = 0.
linsensitivity(paramx, minx, maxx, p5, color = 'r', linestyle = 'dotted')
p5[9] = 0.7 #70% compliance
p5[4] = 0.5
linsensitivity(paramx, minx, maxx, p5, color = 'r', linestyle = 'dashed')
p5[9] = 0.7 #70% compliance
p5[4] = 0.9
linsensitivity(paramx, minx, maxx, p5, color = 'r')
p5[9] = 1. #70% compliance
p5[4] = 0.
linsensitivity(paramx, minx, maxx, p5, color = 'g', linestyle = 'dotted')
p5[9] = 1. #70% compliance
p5[4] = 0.5
linsensitivity(paramx, minx, maxx, p5, color = 'g', linestyle = 'dashed')
p5[9] = 1. #70% compliance
p5[4] = 0.9
linsensitivity(paramx, minx, maxx, p5, color = 'g')
p5[9] = 0. #70% compliance
p5[4] = 0.9
linsensitivity(paramx, minx, maxx, p5, color = 'k')
pp.legend(["70% compliance, 0% VEi","70% compliance, 50% VEi", "70% compliance, 90% VEi", "100% compliance, 0% VEi", "100% compliance, 50% VEi", "100% compliance, 90% VEi", "No vaccine"], bbox_to_anchor=(1.05, 1.0, 0.3, 0.2), loc='lower right')
"""