# Copyright (c) 2012-2023 by the GalSim developers team on GitHub
# https://github.com/GalSim-developers
#
# This file is part of GalSim: The modular galaxy image simulation toolkit.
# https://github.com/GalSim-developers/GalSim
#
# GalSim is free software: redistribution and use in source and binary forms,
# with or without modification, are permitted provided that the following
# conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions, and the disclaimer given in the accompanying LICENSE
#    file.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions, and the disclaimer given in the documentation
#    and/or other materials provided with the distribution.
#

#!/usr/bin/python

# Reads a set of correlation files, calculates the correlations as a function of flux,
# subtracts the intercepts, and plots the result.
# Craig Lage = UC Davis - 30-Jun-16

import matplotlib
matplotlib.use("PDF")
import pyfits as pf
from pylab import *
import sys, glob, time
from scipy import stats
covsteps = 6

numfiles = 5
num_measurements = 1
flux_value = 80000.0 # This is the value (in electrons) we will normalize to
seqnos = [1,2,3,4,5]
numfluxes = len(seqnos)

# Now read in the file names

fluxes = zeros([numfluxes])
covariance = zeros([covsteps, covsteps, numfiles, numfluxes])
reduced_cov = ma.array(zeros([covsteps,covsteps]))
variance = zeros([numfluxes])
gain = zeros([numfluxes])

# Now read in all of the data from the correlations files

for i, seqno in enumerate(seqnos):
    numfluxvalues = 0
    infilename = "correlations_%d.txt"%seqno
    file = open(infilename,'r')
    lines = file.readlines()
    file.close
    for line in lines:
        items = line.split()
        if items[0] == 'ii':
            continue
        try:
             ii = int(items[0])
             jj = int(items[1])
             n = int(items[2])
             if n >= numfiles:
                 break
             covariance[ii,jj,n,i] = float(items[3]) 
             if ii == 0 and jj == 0:
                 fluxes[i] += float(items[4])
                 numfluxvalues += 1
        except Exception as e:
            print "Exception of type %s and args = \n"%type(e).__name__, e.args
            break
    fluxes[i] /= float(numfluxvalues) # calculate the average flux in ADU
    
# Now convert the fluxes into electrons.  To do this we will calculate the gain from the variance
# The calculated gain agrees well with the gain (~4.0-6.0) calculated by other means.
for i in range(numfluxes):
    for ii in range(-5,6):
        for jj in range(-5,6):
            variance[i] += covariance[abs(ii),abs(jj),:,i].mean()
            # This adds in the covariance from surrounding pixels to get the true variance.
    if variance[i] > 1.0E-9:
        gain[i] = 2.0 * fluxes[i] / variance[i]
        print "i = %d, variance = %.2f, gain = %.3f, flux(ADU) = %.2f, flux(e-) = %.2f"%(i,variance[i],gain[i],fluxes[i],fluxes[i]*gain[i])
    else:
        gain[i] = 0.0
    #fluxes[i] *= gain[i] # Don't correct for gain on GalSim simulated flats

# Now plot the covariance vs flux for each pixel, remove the intercept and normalized to a value of flux_value electrons.
# Call this value (calculated at flux_value electrons) the reduced covariance.

outfilename = 'Correlations_vs_Flux_GalSim_18Apr18.pdf'
figure()
subplots_adjust(hspace = 0.5, wspace = 0.8)

for ii in range(covsteps):
    for jj in range(covsteps):
        y = []
        for i in range(numfluxes):
            if variance[i] > 1.0E-9:
                y.append(covariance[ii,jj,:,i].mean() / variance[i])			
        if len(y) == numfluxes:
            slope, intercept, r_value, p_value, std_err = stats.linregress(fluxes[:],y)
            if ii == 0 and jj == 0:
                reduced_cov[ii,jj] = 1.0 + slope * flux_value
            else:
                reduced_cov[ii,jj] = slope * flux_value
            print "ii = %d, jj = %d, slope = %g, reduced_cov = %.4f"%(ii,jj,slope, reduced_cov[ii,jj])
        if ii < 3 and jj < 3:
            plotnum = 3 * ii + jj + 1
            subplot(3,3,plotnum)
            #title("Correlation Coefficient vs Flux ii = %d, jj = %d"%(ii, jj))
            #scatter(fluxes[:], array(y) - intercept)
            try:
                scatter(fluxes[:], y)
                xplot=linspace(0.0, 120000.0, 100)
                #yplot = slope * xplot
                yplot = slope * xplot + intercept
                plot(xplot,yplot)
                xlim(0,120000)
                if ii == 0 and jj == 0:
                    ylim(0.85, 1.05)
                else:
                    ylim(-0.005, 0.015)
                xticks([0, 60000])
                xlabel("Flux(e-)")
                ylabel("Covariance (%d, %d)"%(jj,ii))
            except:
                continue
reduced_cov = ma.masked_where(abs(reduced_cov) < 1.0E-9, reduced_cov)

xvals = []
yvals = []
xfit = []
yfit = []
yerr = []

fullfile = open("corr_meas.txt","w")
fullfile.write("   ii      jj      C      sig\n")

for ii in range(covsteps):
    for jj in range(covsteps):
        n_meas = 1
        # n_meas is the number of good measurements
        cov_mean = reduced_cov[ii,jj]
        cov_std =  0.0#reduced_cov[ii,jj,:,:].std() / sqrt(n_meas)

        rsquared = float(ii*ii + jj*jj)
        if rsquared > 0.1:
            #fullfile.write("i = %d, j = %d, C = %.6f\n"%(jj,ii,reduced_cov[ii,jj]))
            xvals.append(rsquared)
            yvals.append(cov_mean)
            yerr.append(cov_std)
        if rsquared > 1.1 and ii < 3 and jj < 3 and cov_mean > 0.0:
            xfit.append(rsquared)
            yfit.append(cov_mean)

        if ii < 3 and jj < 3:
            plotnum = 3 * ii + jj + 1
            subplot(3,3,plotnum)
            errorbar([flux_value],[cov_mean], yerr = [cov_std] , ls = 'None',marker = '*', ms = 10, color = 'blue')
        fullfile.write("  %d     %d     %.6f     %.6f      %d\n"%(ii, jj, cov_mean, cov_std, n_meas))
legend(bbox_to_anchor=(0.95, 0.5), loc=2, borderaxespad=0., fontsize = 6)

savefig(outfilename)
close()

# Now plot the covariance vs (i^2 + j^2) at flux_value electrons

yvals = array(yvals)
yerr = array(yerr)
ylower = np.maximum(1.1E-5, yvals - yerr)
yerr_lower = yvals - ylower
outfilename = "Correlations_Varying_Flux_GalSim_18Apr18.pdf"
figure()
#title("Correlation Coefficient %d Pairs of Flats - %d Electrons"%(numfiles,flux_value))
xscale('log')
yscale('log')
xlim(0.8,100.0)
ylim(1.0E-5,1.0E-1)
errorbar(xvals,yvals, yerr = [yerr_lower, 2.0*yerr] , ls = 'None',marker = '.', ms = 10, color = 'blue')
slope, intercept, r_value, p_value, std_err = stats.linregress(log10(xfit),log10(yfit))
xplot=linspace(0.0, 2.0, 100)
yplot = slope * xplot + intercept
plot(10**xplot, 10**yplot, color='red', lw = 2, ls = '--')
text(2.5, 0.0030, "Based on %d total flats"%(numfiles*numfluxes*num_measurements*2),fontsize=18)
text(2.5, 0.050, "Slope = %.3f"%slope,fontsize=18)
text(2.5, 0.0232, "C10 = %.4f"%reduced_cov[0,1].mean(),fontsize=18)
text(2.5, 0.0108, "C01 = %.4f"%reduced_cov[1,0].mean(),fontsize=18)
text(2.5, 0.0050, "C11 = %.4f"%reduced_cov[1,1].mean(),fontsize=18)
yticks([1E-4,1E-3,1E-2])
xticks([1.0,10.0,100.0])
fullfile.write("Slope = %.3f\n"%slope)
fullfile.write("C10 = %.5g\n"%reduced_cov[0,1].mean())
fullfile.write("C01 = %.5g\n"%reduced_cov[1,0].mean())
fullfile.close()
xlabel("$i^2 + j^2$",fontsize=16)
ylabel("Covariance",fontsize=18)
savefig(outfilename)
close()


