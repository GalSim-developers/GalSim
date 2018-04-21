# Copyright (c) 2012-2018 by the GalSim developers team on GitHub
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

"""
A script to plot outputs from test_interpolants.py.  The columns-to-quantity relationship is 
hard-coded in, so both of the scripts need to be altered in tandem if the outputs change.
"""
# These two lines allow you to run this script on a terminal with no Display defined, eg through
# a PBS queue
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy
import sys
from textwrap import wrap

def linefit(x,y):
# A routine to return the parameters of a best-fit line (based on least squares method) and the 
# errors on that line in the case where one does not have weights.
# Formulas from Lyons, "Data Analysis for Physical Science Students", ch. 2.4
    xavg = numpy.mean(x)
    yavg = numpy.mean(y)
    xyavg = numpy.mean(x*y)
    xxavg = numpy.mean(x*x)
    n = len(x)
    
    b = (n*xyavg - xavg*yavg)/(n*xxavg - xavg*xavg)
    a = (b*xavg - yavg)/n

    sigsq = 1./(n-2.)*numpy.sum((a+b*x-y)**2)
    sigsqaprime = sigsq/n
    sigsqb = sigsq/numpy.sum((x-xavg)**2)
    sigsqa = sigsqaprime + (xavg)**2*sigsqb
    # B is the slope, A the y-intercept
    return (b, a, numpy.sqrt(sigsqb), numpy.sqrt(sigsqa))

def percentiles(x):
# A routine to return the mean, standard deviation, first and third quartiles, and extrema
# for a one-D list of values.  Quartiles are approximate.  Do not use for rigorous stats tests.
    n = len(x)
    mean = numpy.mean(x)
    stdev = numpy.std(x)
    sortx = numpy.sort(x)
    # Next two formulas are exact for the case where the quartiles are single unambiguous data
    # points--probably close enough for other purposes.
    quarter = max(0,int(round((n-3.)/4.))) 
    threequarter = min(2+int(round(3.*(n-3.)/4.)),n-1) 
    # Return things in this bizarre fashion so numpy treats them as 1-element arrays
    # and not as scalars (makes matplotlib blow up)
    return (numpy.array((mean,)), stdev, numpy.array((mean-sortx[quarter],)),  
            numpy.array((sortx[threequarter]-mean,)), 
            numpy.array((mean-min(x),)), numpy.array((max(x)-mean,)))
        
# plotfile should be the output from test_interpolants.py, filenamebase the root of the .png files
# this script will output, to which something like "_x_dg1_g1.png" will be appended
def plot_interpolants(plotfile, filenamebase):
    interpolant_titles = ['cubic', 'quintic', 
                          'lanczos3', 'lanczos4', 'lanczos5', 'lanczos7', 'default']
    # Luminosity-matched colors for screen presentations
#    interpolant_colors = ['#7c7c7c', '#000000', '#db00db', '#6767ff', '#008989', 
#                          '#009000', '#808000', '#d35400', '#f80000']
    interpolant_colors = ['#7c7c7c', '#000000', '#db00db', '#6767ff',  
                          '#808000', '#f80000']
    padding_titles = ['pad4','pad6']
    padding_shapes = ['o', 's']

    # Begin by plotting data from the file containing the info with shear, rotation, etc.
    plotdata = numpy.loadtxt(plotfile)
    # Remove measurement failures
    no_result_mask = ((abs(plotdata[:,10]+10)>5.5) & (abs(plotdata[:,13]+10)>5.5) &
                      (abs(plotdata[:,11]+10)>5.5) & (abs(plotdata[:,14]+10)>5.5) &
                      (abs(plotdata[:,11]+plotdata[:,12]+10)>3) & 
                      (abs(plotdata[:,14]+plotdata[:,15]+10)>3) &
                      (plotdata[:,16]>0) & (plotdata[:,17]>0) & (plotdata[:,14]+plotdata[:,15]>0))
    plotdata = plotdata[no_result_mask]
    # Make lists of the various tested quantities
    x_interpolants = list(set(plotdata[:,1]))
    x_interpolants.sort()
    x_interpolants = [int(i) for i in x_interpolants]
    k_interpolants = list(set(plotdata[:,2]))
    k_interpolants.sort()
    k_interpolants = [int(i) for i in k_interpolants]
    interpolants = list(set(x_interpolants) | set(k_interpolants))
    paddings = list(set(plotdata[:,3]))
    paddings.sort()
    paddings = [int(i) for i in paddings]
    # And give some of the columns more sensible names, for better human readability
    intrinsic_g1 = plotdata[:,10]
    intrinsic_g2 = plotdata[:,13]
    intrinsic_size = plotdata[:,16]
    expected_g1 = plotdata[:,11]
    expected_g2 = plotdata[:,14]
    expected_size = plotdata[:,17]
    expected_g1_difference = expected_g1 - intrinsic_g1
    expected_g2_difference = expected_g2 - intrinsic_g2
    g1_difference = plotdata[:,12]
    g2_difference = plotdata[:,15]
    frac_size_difference = plotdata[:,18]
    applied_shears_g1 = plotdata[:,4]
    applied_shears_g2 = plotdata[:,5]
    applied_magnification = plotdata[:,6]        
    applied_magnification_list = list(set(applied_magnification))
    applied_magnification_list.sort()
    applied_angle = plotdata[:,7]
    applied_shiftx = plotdata[:,8]
    applied_shifty = plotdata[:,9]
    # Construct masks for the different changes applied to the data.
    # This had some trouble with == 0, and all applied shears/mags are >1.E-5, so
    # now use that as a cutoff.
    g1_only = ((abs(applied_shears_g1)>=1.E-5) & (abs(applied_shears_g2)<1.E-5) & 
                (abs(applied_magnification-1)<1.E-5))
    g2_only = ((abs(applied_shears_g1)<1.E-5) & (abs(applied_shears_g2)>=1.E-5) & 
                (abs(applied_magnification-1)<1.E-5))
    magnification_only = ((abs(applied_shears_g1)<1.E-5) & (abs(applied_shears_g2)<1.E-5) & 
                            (abs(applied_magnification-1)>=1.E-5))
    shift2_only = ((abs(applied_shiftx)>=1.E-5) & (abs(applied_shifty)>=1.E-5))
    shift1_only = ((abs(applied_shiftx)>=1.E-5) | (abs(applied_shifty)>=1.E-5)) # any shift
    shift1_only = (shift1_only & numpy.logical_not(shift2_only)) # remove 45-degree shifts
    angle_only = ((abs(applied_shears_g1)<1.E-5) & (abs(applied_shears_g2)<1.E-5) & 
                    (abs(applied_magnification-1)<1.E-5) & (abs(applied_shiftx)<1.E-5) &
                    (abs(applied_shifty)<1.E-5))
    allthree = ((abs(applied_shears_g1)>=1.E-5) & (abs(applied_shears_g2)>=1.E-5) & 
                (abs(applied_magnification-1)>=1.E-5))


    subinterpolant_titles=["x interpolants","k interpolants"]
    subinterpolant_indices=[1,2]
    # Mask for k or x interpolants: 9 is default, so != 9 is a changed parameter
    subinterpolant_masks = [abs(plotdata[:,i]-9)>1.E-5 for i in subinterpolant_indices]
    interpolant_masks = [(abs(plotdata[:,1]-i)<1.E-5) | (abs(plotdata[:,2]-i)<1.E-5) 
                                                                    for i in interpolants]
    pad_masks = [abs(plotdata[:,3]-i)<1.E-5 for i in paddings]
    
    for subinterp_index, subinterp_title, subinterp_mask in zip(
                subinterpolant_indices,subinterpolant_titles,subinterpolant_masks):
        combomask = [g1_only, g2_only, magnification_only, allthree, shift1_only, shift2_only] 
        combomask_titles = ["only G1", "only G2", "only magnification", "G1+G2+magnification",
                            "90 degree shifts", "45 degree shifts"]
        combomask_filetitles = ["change-g1", "change-g2", "change-mag", "change-all", 
                                "change-shift1", "change-shift2"]
        for cmask, ctitle, octitle in zip(combomask,combomask_titles,combomask_filetitles):
            mask = [[cmask & subinterp_mask & p & i for p in pad_masks] 
                                                for i in interpolant_masks]
            data_types = [(expected_g1, g1_difference, "delta_g1_as_f_of_expected_g1",
                                "delta g1 as a function of expected g1"), 
                          (expected_g1_difference, g1_difference, 
                                "delta_g1_as_f_of_expected_minus_intrinsic_g1",
                                "delta g1 as a function of (expected - intrinsic) g1"),
                          (applied_shears_g1, g1_difference, "delta_g1_as_f_of_applied_g1",
                                "delta g1 as a function of applied g1"), 
                          (applied_shears_g2, g1_difference, "delta_g1_as_f_of_applied_g2",
                                "delta g1 as a function of applied g2"), 
                          (expected_g2, g2_difference, "delta_g2_as_f_of_expected_g2",
                                "delta g2 as a function of expected g2"), 
                          (expected_g2_difference, g2_difference, 
                                "delta_g2_as_f_of_expected_minus_intrinsic_g2",
                                "delta g2 as a function of (expected - intrinsic) g2"),
                          (applied_shears_g2, g2_difference, "delta_g2_as_f_of_applied_g2",
                                "delta g2 as a function of applied g2"), 
                          (applied_shears_g1, g2_difference, "delta_g2_as_f_of_applied_g1",
                                "delta g2 as a function of applied g1"),  
                          (expected_size, frac_size_difference, 
                                "delta_frac_size_as_f_of_expected_size", 
                                "delta size/expected size as a function of expected size"),
                          (applied_magnification, frac_size_difference, 
                                "delta_frac_size_as_f_of_applied_magnification", 
                                "delta size/expected size as a function of applied magnification")]

            # Plot m vs pad_factor for the various interpolant + added shear permutations
            for x,y,optitle,ptitle in data_types:
                ymax = 0.0002
                ymin = -0.0002
                writeplot = False # In case none of the masks pass our plotability tests
                ixoffset = -0.5*len(interpolants) # to offset for clarity
                xoffset = 2/30.
                plt.axhline(0.,color='black')
                for i in range(len(interpolants)-1):
                    for j in range(len(paddings)):
                        mx = x[mask[i][j]]
                        if len(mx)>0 and max(mx)-min(mx)>0.05:
                            writeplot = True
                            m,c,merr,cerr = linefit(mx,y[mask[i][j]])
                            while m>ymax or m<ymin:
                                ymax*=2
                                ymin*=2
                            plt.errorbar(paddings[j]+ixoffset*xoffset,m,yerr=merr,
                                color=interpolant_colors[interpolants[i]],)
                            if j==1:
                                plt.plot(paddings[j]+ixoffset*xoffset,m,
                                    marker='o',
                                    color=interpolant_colors[interpolants[i]],
                                    label=interpolant_titles[interpolants[i]])
                            else:
                                plt.plot(paddings[j]+ixoffset*xoffset,m,
                                    marker='o',
                                    color=interpolant_colors[interpolants[i]])
                    ixoffset+=1
                plt.xlabel('pad_factor')
                plt.ylabel('m')
                plt.title('\n'.join(wrap(subinterp_title+', '+ptitle+', '+ctitle,60)))
                plt.xlim([3.5,8]) # so the legend doesn't overlap
                plt.tight_layout()
                if "frac_size" not in optitle:
                    if ymax>0.0002:
                        plt.axhline(-0.0002,color='silver',linestyle='--')
                        plt.axhline(0.0002,color='silver',linestyle='--')
                    plt.ylim((ymin,ymax))
                plt.legend()
                if writeplot:
                    plt.savefig(filenamebase+'.'+subinterp_title[0]+'.'+optitle+'.'+octitle+'.png')
                plt.clf()

            # Plot average errors
            data_types = [(g1_difference, "g1", "g1_errors",
                                "average g1 errors"), 
                          (g2_difference, "g2", "g2_errors",
                                "average g2 errors"), 
                          (frac_size_difference, "size/expected size", "frac_size_errors",
                                "average fractional size errors")]
            # Plot average error vs pad_factor for the various interpolant + 
            # added shear permutations
            for x,ytitle,optitle,ptitle in data_types:
                ixoffset = -0.5*len(interpolants) # to offset for clarity
                xoffset = 2/30.
                for i in range(len(interpolants)-1):
                    for j in range(len(paddings)):
                        mx = x[mask[i][j]]
                        datapoint = [numpy.average(numpy.abs(mx))]
                        if j==1:
                            plt.plot(paddings[j]+ixoffset*xoffset,datapoint,
                                marker='o',
                                color=interpolant_colors[interpolants[i]],
                                label=interpolant_titles[interpolants[i]])
                        else:
                            plt.plot(paddings[j]+ixoffset*xoffset,datapoint,
                                marker='o',
                                color=interpolant_colors[interpolants[i]])
                    ixoffset+=1
                plt.xlabel('pad_factor')
                plt.ylabel('<|delta '+ytitle+'|>')
                plt.title('\n'.join(wrap(subinterp_title+', '+ptitle+', '+ctitle,60)))
                plt.xlim([3.5,8]) # so the legend doesn't overlap
                plt.legend()
                plt.tight_layout()
                plt.savefig(filenamebase+'.'+subinterp_title[0]+'.'+optitle+'.'+octitle+'.png')
                plt.clf()


        # Plot differentials as a function of angles
        data_types = [(applied_angle, g1_difference, "applied angle", "delta g1",
                         "delta_g1_as_f_of_applied_angle"), 
                        (applied_angle, g2_difference, "applied angle", "delta g2",
                         "delta_g2_as_f_of_applied_angle"), 
                        (applied_angle, frac_size_difference, "applied angle", 
                         "delta size/expected size", "delta_frac_size_as_f_of_applied_angle")]
        
        for x,y,xtitle,ytitle,optitle in data_types:
            for j in range(len(paddings)):
                bins = list(set(applied_angle))
                bins.sort()
                # Plot quartiles for each dimension
                histmask = [[[(abs(x-bin)<1.E-5) & angle_only & subinterp_mask & p & i 
                                                    for bin in bins] for p in pad_masks] 
                                                    for i in interpolant_masks]
                xoffset = (bins[1]-bins[0])/100.
                ixoffset = int(-0.5*len(interpolants)*len(paddings))
                for i in range(len(interpolants)-1):
                    for k in range(len(bins)):
                        ty = y[histmask[i][j][k]]
                        if len(ty)>0:
                            mean, stdev, q1, q3, ymin, ymax = percentiles(y[histmask[i][j][k]])
                            plt.errorbar(bins[k]+ixoffset*xoffset, mean, yerr=[q1,q3],
                                            color=interpolant_colors[interpolants[i]],
                                            linestyle='dashed')
                            plt.errorbar(bins[k]+ixoffset*xoffset, mean, yerr=[ymin,ymax],
                                            color=interpolant_colors[interpolants[i]])
                            if k==0:
                                plt.plot(bins[k]+ixoffset*xoffset, mean, 
                                            color=interpolant_colors[interpolants[i]],
                                            marker=padding_shapes[j],
                                            label=interpolant_titles[interpolants[i]])
                            else:
                                plt.plot(bins[k]+ixoffset*xoffset, mean, 
                                            color=interpolant_colors[interpolants[i]],
                                            marker=padding_shapes[j])
                        ixoffset+=1
                plt.xlabel(xtitle)
                plt.ylabel(ytitle)
                plt.xlim([-10,250])
                plt.title('\n'.join(wrap(subinterp_title+', '+padding_titles[j]+', '
                            +'quartiles, min and max by angle',60)))
                plt.legend()
                plt.savefig(filenamebase+'.'+subinterp_title[0]+'.'+optitle+
                            '.pad'+str(paddings[j])+'.png', bbox_inches='tight')
                plt.clf()
            
if __name__=='__main__':
    if len(sys.argv)<3:
        print "To use this script, call "
        print "./plot_test_interpolants.py name-of-file-to-plot root-of-image-file-names"
    else:   
        plot_interpolants(sys.argv[-2], sys.argv[-1])
