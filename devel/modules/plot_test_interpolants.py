import matplotlib.pyplot as plt
import numpy
from matplotlib.backends.backend_pdf import PdfPages

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
        

def main():
    interpolant_titles = ['nearest', 'sinc', 'linear', 'cubic', 'quintic', 
                          'lanczos3', 'lanczos4', 'lanczos5', 'lanczos7', 'default']
    # Luminosity-matched colors for screen presentations
    interpolant_colors = ['#7c7c7c', '#000000', '#db00db', '#6767ff', '#008989', 
                          '#009000', '#808000', '#d35400', '#f80000']
    padding_titles = ['pad2','pad4','pad6']
    padding_shapes = ['o', 's', '^']

    # Begin by plotting data from the file containing the info with shear, rotation, etc.
#    plotfiles = ['interpolant_test_output_ground.dat', 'interpolant_test_output_space.dat',
#                 'interpolant_test_output_original.dat']
    plotfiles=['interpolant_test_output_original.dat']
    for plotfile in plotfiles:
        print plotfile,
        plotdata = numpy.loadtxt(plotfile)
        # Tally measurement failures (which appear as values==-10)
        no_result_measurement_mask = ((abs(plotdata[:,8]+plotdata[:,9]+10)<1.E-5) | 
                                      (abs(plotdata[:,10]+plotdata[:,11]+10)<1.E-5))  
        measurement_failures = plotdata[no_result_measurement_mask]
        measurement_failures_shear_bygal = [
            len(measurement_failures[abs(measurement_failures[10])<1.E-5,0]==i) 
                                                                for i in range(100)]
        measurement_failures_angle_bygal = [
            len(measurement_failures[abs(measurement_failures[10])>1.E-5,0]==i) 
                                                                for i in range(100)]
        measurement_failures_shear_byint = [
            len(measurement_failures[abs(measurement_failures[10])<1.E-5,0]==i)  
                                                                for i in range(10)]
        measurement_failures_angle_byint = [
            len(measurement_failures[abs(measurement_failures[10])>1.E-5,0]==i)  
                                                                for i in range(10)]
        measurement_failures_shear_byint = measurement_failures_shear_byint[:9] # remove "default"
        measurement_failures_angle_byint = measurement_failures_angle_byint[:9]
        no_result_original_mask = ((abs(plotdata[:100,8]+10)<1.E-5) | 
                                   (abs(plotdata[:100,10]+10)<1.E-5))
        original_failures = plotdata[no_result_original_mask,0]

        del no_result_measurement_mask
        del measurement_failures
        del no_result_original_mask

        # Remove measurement failures
        no_result_mask = ((abs(plotdata[:,8]+10)>1.E-5) & (abs(plotdata[:,8]+10)>1.E-5) &
                          (abs(plotdata[:,8]+plotdata[:,9]+10)>1.E-5) & 
                          (abs(plotdata[:,10]+plotdata[:,11]+10)>1.E-5))  
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
        expected_g1 = plotdata[:,8]
        expected_g2 = plotdata[:,10]
        expected_size = plotdata[:,12]
        g1_difference = plotdata[:,9]
        g2_difference = plotdata[:,11]
        frac_size_difference = plotdata[:,13]
        applied_shears_g1 = plotdata[:,4]
        applied_shears_g2 = plotdata[:,5]
        applied_magnification = plotdata[:,6]        
        applied_magnification_list = list(set(applied_magnification))
        applied_magnification_list.sort()
        applied_angle = plotdata[:,7]
        # Construct masks for the different changes applied to the data.
        # This had some trouble with == 0, and all applied shears/mags are >1.E-5, so
        # now use that as a cutoff.
        g1_only = ((abs(applied_shears_g1)>=1.E-5) & (abs(applied_shears_g2)<1.E-5) & 
                   (abs(applied_magnification-1)<1.E-5))
        g2_only = ((abs(applied_shears_g1)<1.E-5) & (abs(applied_shears_g2)>=1.E-5) & 
                   (abs(applied_magnification-1)<1.E-5))
        magnification_only = ((abs(applied_shears_g1)<1.E-5) & (abs(applied_shears_g2)<1.E-5) & 
                              (abs(applied_magnification-1)>=1.E-5))
        angle_only = ((abs(applied_shears_g1)<1.E-5) & (abs(applied_shears_g2)<1.E-5) & 
                      (abs(applied_magnification-1)<1.E-5))
        allthree = ((abs(applied_shears_g1)>=1.E-5) & (abs(applied_shears_g2)>=1.E-5) & 
                    (abs(applied_magnification-1)>=1.E-5))
        combomask = [g1_only, g2_only, magnification_only] # These are plotted the same
        combomask_titles = ["only G1", "only G2", "only magnification"]
        combomask_filetitles = ["g1", "g2", "mag"]
        # File name roots
        if 'ground' in plotfile:
            filenamebase = 'interpolant_test_ground_'
        elif 'original' in plotfile:
            filenamebase = 'interpolant_test_original_'
        else:
            filenamebase = 'interpolant_test_space_'
        
        subinterpolant_titles=["x interpolants","k interpolants"]
        subinterpolant_indices=[1,2]
        # Mask for k or x interpolants: 9 is default, so != 9 is a changed parameter
        subinterpolant_masks = [abs(plotdata[:,i]-9)>1.E-5 for i in subinterpolant_indices]
        interpolant_masks = [(abs(plotdata[:,1]-i)<1.E-5) | (abs(plotdata[:,2]-i)<1.E-5) 
                                                                     for i in interpolants]
        pad_masks = [abs(plotdata[:,3]-i)<1.E-5 for i in paddings]
        for subinterp_index, subinterp_title, subinterp_mask in zip(
                    subinterpolant_indices,subinterpolant_titles,subinterpolant_masks):
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
                        ixoffset+=1
                    plt.xlabel('pad_factor')
                    plt.ylabel('m')
                    plt.title(subinterp_title+', '+ptitle+', '+ctitle)
                    plt.xlim([1.5,9]) # so the legend doesn't overlap
                    plt.legend()
                    plt.savefig(filenamebase+subinterp_title[0]+'_'+optitle+'_'+octitle+'.png')
                    plt.clf()

                # Plot m as a function of inputs in OTHER dimensions
                if ctitle == "only G1":    
                    data_types = [(applied_shears_g1, g2_difference, "dg2vg1",
                                        "delta g2 as a function of applied g1"), 
                                  (applied_shears_g1, frac_size_difference, "dsizevg1",
                                        "delta size/expected size as a function of applied g1")]
                elif ctitle == "only G2":
                    data_types = [(applied_shears_g2, g1_difference, "dg1vg2",
                                        "delta g1 as a function of applied g2"), 
                                  (applied_shears_g2, frac_size_difference, "dsizevg2",
                                        "delta size/expected size as a function of applied g2")]
                else:
                    data_types = [(applied_magnification, g1_difference, "dg1vsize",
                                        "delta g1 as a function of applied magnification"), 
                                  (applied_magnification, g2_difference, "dg2vsize",
                                        "delta g2 as a function of applied magnification")]
                
                for x,y,optitle, ptitle in data_types:
                    ixoffset = int(-0.5*len(interpolants))
                    for i in range(len(interpolants)-1):
                        xoffset = 2/20.
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
                        ixoffset+=1
                    plt.xlabel('pad_factor')
                    plt.ylabel('m')
                    plt.xlim([1.5,9])
                    plt.title(subinterp_title+', '+ptitle+', '+ctitle)
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
                    ixoffset+=1
                plt.xlabel('pad_factor')
                plt.ylabel('m')
                plt.xlim([1.5,9])
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
                    plt.xlim([-10,200])
                    plt.title(subinterp_title+', '+padding_titles[j]+', '
                              +'stdev, quartiles, min and max by angle')
                    plt.legend()
                    plt.savefig(filenamebase+subinterp_title[0]+'_'+optitle+'_pad'+str(j)+'_angle.png')
                    plt.clf()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        if len(original_failures) == 100:
            rects1 = ax.bar(numpy.array(range(0,300,3)), original_failures, 1, color='g')
            rects2 = ax.bar(numpy.array(range(1,300,3)), measurement_failures_shear_bygal, 
                            1, color='g')
            rects3 = ax.bar(numpy.array(range(2,300,3)), measurement_failures_angle_bygal, 
                            1, color='b')
            ax.set_xticks(numpy.array(range(0,300,3)+1.5))
            ax.legend( (rects1[0], rects2[0], rects3[0]), 
                       ('Original galaxy', 'applied shear', 'applied angle') )
        else:
            rects1 = ax.bar(numpy.array(range(0,200,2)), measurement_failures_shear_bygal, 
                            1, color='g')
            rects2 = ax.bar(numpy.array(range(1,200,2)), measurement_failures_angle_bygal, 
                            1, color='b')
            ax.set_xticks(numpy.array(range(0,200,2))+0.5)
            ax.legend( (rects1[0], rects2[0]), ('applied shear', 'applied angle') )
        ax.set_ylabel('Number of failures')
        ax.set_xlabel('RealGalaxy indices')
        ax.set_title('Measurement failures by index')
        ax.set_xticklabels( numpy.array(range(100)) )
        plt.savefig('indexfailures.png')

        fig = plt.figure()
        ax = fig.add_subplot(111)
        rects1 = ax.bar(range(len(interpolants)), measurement_failures_shear_byint, 1, color='r')
        rects2 = ax.bar(range(len(interpolants)), measurement_failures_angle_byint, 1, color='r')
        ax.set_ylabel('Number of failures')
        ax.set_xlabel('Interpolants')
        ax.set_title('Measurement failures by interpolant')
        ax.set_xticks(numpy.array(range(0,20,2))+0.5)
        ax.set_xticklabels( [interpolant_titles[i] for i in interpolants] )
        plt.savefig('interpolantfailures.png')
                    

if __name__=='__main__':
    main()
