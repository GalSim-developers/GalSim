# Copyright (c) 2012-2020 by the GalSim developers team on GitHub
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


# Latest version of correlation calculation based on Johann Cohen-Tanugi code
# This version analyzes GalSim generated flats
# Craig Lage 18-Apr-18

import pyfits as pf
from pylab import *
import os, sys, glob, time
import numpy as np


edges = np.zeros([2000,509],dtype=bool)
xmin = 5
xmax = 504
ymin = 5
ymax =1995
edges[0:ymin,:] = True
edges[ymax:-1,:] = True
edges[:,0:xmin] = True
edges[:,xmax:-1] = True
edge_mask = np.ma.make_mask(edges)
print edges.sum()
mydir = "/Users/cslage/Research/LSST/code/galsim-developers-18apr18/GalSim/devel/lsst/flats/"
series = int(sys.argv[1])
even_patterns = sort(glob.glob(mydir+'flat_%d_*.fits*'%series))[0::2]
odd_patterns = sort(glob.glob(mydir+'flat_%d_*.fits*'%series))[1::2]
print "There are %d even patterns and %d oddpatterns"%(len(even_patterns), len(odd_patterns))
sys.stdout.flush()
numfiles = len(even_patterns)

filename = 'correlations_%d.txt'%series
file = open(filename,'w')
line = 'ii     jj     n     extname     covariance     median\n'
file.write(line)
file.close()
for n in range(numfiles):
    time_start=time.time()
    ccd1=even_patterns[n]
    ccd2=odd_patterns[n]
    image1 = pf.getdata(even_patterns[n])
    image2 = pf.getdata(odd_patterns[n])        
    avemed  = (np.median(image1) + np.median(image2)) / 2.0
    #subtract the two images
    fdiff = image1 - image2
    ncols,nrows=509,2000
    # Create a simple mask based on thresholding the variance
    Nsigma=6
    diffmed  = np.median(fdiff)
    thresh = Nsigma*np.std(fdiff)
    fdiff_mask = np.ma.masked_where(abs( fdiff-diffmed ) > thresh,fdiff,copy=True)
    fdiff_mask.mask = np.ma.mask_or(fdiff_mask.mask,edges)
    #print "diffmed = %f, diffmean = %f, vardiff = %f, sigdiff = %f"%(diffmed, np.mean(fdiff_mask), np.var(fdiff_mask), np.std(fdiff_mask))
    #print "diffmax = %f, diffmin = %f"%(fdiff.min(), fdiff.max())
    #print fdiff[38,20:30]
    #sys.exit()
    #print "For file %d, segment = %s,image1 median = %.2f, image median = %.2f, diff median = %.2f, diffmean = %.2f"%(n,extname, np.median(image1.getArrays()[0]), np.median(image2.getArrays()[0]), diffmed, np.mean(fdiff_mask.data))
    # Finally compute the pixel covariance
    # Clearly the code is not good here, as the nested loop take time.
    # Proof of concept : set k and l to fixed values:
    k=1
    l=1
    i_range = range(k,ncols)
    j_range = range(l,nrows)

    for k in range(0,6):
        for l in range(0,6):
            npixused1=0
            if k==0 and l==0:
                corr = np.var(fdiff_mask)
            else:
                temp1=fdiff_mask[l:nrows  ,   k:ncols]
                data1=temp1.data
                mask1=temp1.mask
                temp2=fdiff_mask[0:nrows-l  , 0:ncols-k]
                data2=temp2.data
                mask2=temp2.mask
                or_mask=np.logical_or(mask1,mask2)
                npixused1=or_mask.size-or_mask.sum()
                sub1=np.ma.MaskedArray(data1,or_mask)
                sub2=np.ma.MaskedArray(data2,or_mask)
                sum11=sub1.sum()
                sum21=sub2.sum()
                sum121=(sub1*sub2).sum()
                corr = (sum121 - sum11*sum21/npixused1)/npixused1

            print "For file %d, NumDataPoints = %d, ii = %d, jj = %d,Cij = %.2f"%(n, npixused1, l, k, corr)
            file = open(filename,'a')
            if k==0 and l==0:
                line = '%d     %d     %d      %f      %f\n'%(l,k,n,corr,avemed)
            else:
                line = '%d     %d     %d      %f\n'%(l,k,n,corr)
            file.write(line)
            file.close()
    time_finish = time.time()
    elapsed = time_finish - time_start
    print "Elapsed time for file %d  = %.2f"%(n,elapsed)
    sys.stdout.flush()

