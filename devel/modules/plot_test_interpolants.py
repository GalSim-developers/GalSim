#!/opt/python27/bin/python2.7

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
    interpolant_titles = ['nearest', 'sinc', 'linear', 'cubic', 'quintic', 
                          'lanczos3', 'lanczos4', 'lanczos5', 'lanczos7', 'default']
    # Luminosity-matched colors for screen presentations
    interpolant_colors = ['#7c7c7c', '#000000', '#db00db', '#6767ff', '#008989', 
                          '#009000', '#808000', '#d35400', '#f80000']
    padding_titles = ['pad2','pad4','pad6']
    padding_shapes = ['o', 's', '^']

    # Begin by plotting data from the file containing the info with shear, rotation, etc.
    plotdata = numpy.loadtxt(plotfile)
    # Remove measurement failures
    no_result_mask = ((abs(plotdata[:,10]+10)>5.5) & (abs(plotdata[:,12]+10)>5.5) &
                        (abs(plotdata[:,10]+plotdata[:,11]+10)>3) & 
                        (abs(plotdata[:,12]+plotdata[:,13]+10)>3) &
                        (plotdata[:,14]>0) & (plotdata[:,14]+plotdata[:,15]>0))  
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
    expected_g1 = plotdata[:,10]
    expected_g2 = plotdata[:,12]
    expected_size = plotdata[:,14]
    g1_difference = plotdata[:,11]
    g2_difference = plotdata[:,13]
    frac_size_difference = plotdata[:,15]
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
        combomask = [g1_only, g2_only, magnification_only] # These are plotted the same
        combomask_titles = ["only G1", "only G2", "only magnification"]
        combomask_filetitles = ["g1", "g2", "mag"]
        for cmask, ctitle, octitle in zip(combomask,combomask_titles,combomask_filetitles):
            mask = [[cmask & subinterp_mask & p & i for p in pad_masks] 
                                                for i in interpolant_masks]
            data_types = [(expected_g1, g1_difference, "dg1",
                                "delta g1 as a function of expected g1"), 
                            (expected_g2, g2_difference, "dg2",
                                "delta g2 as a function of expected g2"), 
                            (expected_size, frac_size_difference, "dsize", 
                                "delta size/expected size as a function of expected size")]

            # Plot m vs pad_factor for the various interpolant + added shear permutations
            for x,y,optitle,ptitle in data_types:
                ixoffset = int(-0.5*len(interpolants)) # to offset for clarity
                xoffset = 2/20.
                plt.axhline(0.,color='black')
                for i in range(len(interpolants)-1):
                    for j in range(len(paddings)):
                        m,c,merr,cerr = linefit(x[mask[i][j]],y[mask[i][j]])
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
                        if j==1 and i==len(interpolants)-2:
                            ymean = m
                    ixoffset+=1
                plt.xlabel('pad_factor')
                plt.ylabel('m')
                plt.title(subinterp_title+', '+ptitle+', '+ctitle)
                plt.xlim([1.5,9]) # so the legend doesn't overlap
                if numpy.any(x!=expected_size):
                    plt.ylim([-0.0005,+0.0005])
                plt.legend()
                plt.savefig(filenamebase+subinterp_title[0]+'_'+optitle+'_'+octitle+'.png')
                plt.clf()
        # Plot m and c in the case where we've added all 3 components
        cmask = allthree
        ctitle = "G1+G2+magnification"
        octitle = "all"
        mask = [[cmask & subinterp_mask & p & i for p in pad_masks] for i in interpolant_masks]
        data_types = [(expected_g1, g1_difference, "dg1", 
                                "delta g1 as a function of expected g1"), 
                        (expected_g2, g2_difference, "dg2", 
                                "delta g2 as a function of expected g2"), 
                        (expected_size, frac_size_difference, "dsize", 
                                "delta size/expected size as a function of expected size")]
            
        for x,y,optitle,ptitle in data_types:
            ixoffset = int(-0.5*len(interpolants))
            for i in range(len(interpolants)-1):
                xoffset = 2/20.
                plt.axhline(0.,color='black')
                for j in range(len(paddings)):
                    m,c,merr,cerr = linefit(x[mask[i][j]],y[mask[i][j]])
                    plt.errorbar(paddings[j]+ixoffset*xoffset,m,yerr=merr,
                        color=interpolant_colors[interpolants[i]],)
                    if j==1:
                        plt.plot(paddings[j]+ixoffset*xoffset,m,
                            marker='o',
                            color=interpolant_colors[interpolants[i]],
                            label=interpolant_titles[interpolants[i]])
                        if i==len(interpolants)-2:
                            ymean = m
                    else:
                        plt.plot(paddings[j]+ixoffset*xoffset,m,
                            marker='o',
                            color=interpolant_colors[interpolants[i]])
                ixoffset+=1
            plt.xlabel('pad_factor')
            plt.ylabel('m')
            plt.xlim([1.5,9])
            if numpy.any(x!=expected_size):
                plt.ylim([-0.0005,+0.0005])
            plt.title(subinterp_title+', '+ptitle+', '+ctitle)
            plt.legend()
            plt.savefig(filenamebase+subinterp_title[0]+'_'+optitle+'_'+octitle+'.png')
            plt.clf()
        
        # Plot m and c for the different shift cases
        combomask = [shift1_only,shift2_only] # These are plotted the same
        combomask_titles = ["90 degree shifts", "45 degree shifts"]
        combomask_filetitles = ["shift1", "shift2"]
        for cmask, ctitle, octitle in zip(combomask,combomask_titles,combomask_filetitles):
            mask = [[cmask & subinterp_mask & p & i for p in pad_masks] 
                                                for i in interpolant_masks]
            data_types = [(expected_g1, g1_difference, "dg1", 
                                "delta g1 as a function of expected g1"), 
                            (expected_g2, g2_difference, "dg2", 
                                "delta g2 as a function of expected g2"), 
                            (expected_size, frac_size_difference, "dsize", 
                                "delta size/expected size as a function of expected size")]
            
            for x,y,optitle,ptitle in data_types:
                ixoffset = int(-0.5*len(interpolants))
                for i in range(len(interpolants)-1):
                    xoffset = 2/20.
                    plt.axhline(0.,color='black')
                    for j in range(len(paddings)):
                        m,c,merr,cerr = linefit(x[mask[i][j]],y[mask[i][j]])
                        plt.errorbar(paddings[j]+ixoffset*xoffset,m,yerr=merr,
                            color=interpolant_colors[interpolants[i]],)
                        if j==1:
                            plt.plot(paddings[j]+ixoffset*xoffset,m,
                                marker='o',
                                color=interpolant_colors[interpolants[i]],
                                label=interpolant_titles[interpolants[i]])
                            if i==len(interpolants)-2:
                                ymean = m
                        else:
                            plt.plot(paddings[j]+ixoffset*xoffset,m,
                                marker='o',
                                color=interpolant_colors[interpolants[i]])
                    ixoffset+=1
                plt.xlabel('pad_factor')
                plt.ylabel('m')
                plt.xlim([1.5,9])
                plt.ylim([-0.0005,+0.0005])
                plt.title(subinterp_title+', '+ptitle+', '+ctitle)
                plt.legend()
                plt.savefig(filenamebase+subinterp_title[0]+'_'+optitle+'_'+octitle+'.png')
                plt.clf()

        # Plot differentials as a function of angles
        data_types = [(applied_angle, g1_difference, "applied angle", "delta g1", "dg1"), 
                        (applied_angle, g2_difference, "applied angle", "delta g2", "dg2"), 
                        (applied_angle, frac_size_difference, "applied angle", 
                                                    "delta size/expected size", "dsize")]
        
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
                            plt.errorbar(bins[k]+ixoffset*xoffset, mean, yerr=stdev,
                                            color=interpolant_colors[interpolants[i]],
                                            linestyle='dashed')
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
                plt.title(subinterp_title+', '+padding_titles[j]+', '
                            +'stdev, quartiles, min and max by angle')
                plt.legend()
                plt.savefig(filenamebase+subinterp_title[0]+'_'+optitle+'_pad'+str(j)+'_angle.png')
                plt.clf()
            
if __name__=='__main__':
    if len(sys.argv)<3:
        print "To use this script, call "
        print "./plot_test_interpolants.py name-of-file-to-plot root-of-image-file-names"
    else:   
        plot_interpolants(sys.argv[-2], sys.argv[-1])
