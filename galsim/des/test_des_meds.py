# Copyright 2012, 2013 The GalSim developers:
# https://github.com/GalSim-developers
#
# This file is part of GalSim: The modular galaxy image simulation toolkit.
#
# GalSim is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# GalSim is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with GalSim.  If not, see <http://www.gnu.org/licenses/>

# Requirements:
# meds  - https://github.com/esheldon/meds
# fitso - https://github.com/esheldon/fitsio


import numpy as np
import os
import sys

try:
    import galsim
except ImportError:
    path, filename = os.path.split(__file__)
    sys.path.append(os.path.abspath(os.path.join(path, "..")))
    import galsim

import numpy

def test_meds():
    """
    Create two objects, each with two exposures. Save them to a MEDS file. 
    Load the MEDS file. Compare the created objects with the one read by MEDS.
    """

    # initialise empty MultiExposureObject list
    objlist = []

    # first obj
    img11 = numpy.ones([32,32])*111
    img12 = numpy.ones([32,32])*112
    seg11 = numpy.ones([32,32])*121
    seg12 = numpy.ones([32,32])*122
    wth11 = numpy.ones([32,32])*131
    wth12 = numpy.ones([32,32])*132
    jac11 = numpy.array([[11.1 , 11.2],[12.3 , 11.4]])
    jac12 = numpy.array([[12.1 , 12.2],[12.3 , 12.4]])

    # create lists
    images =  [img11,img12]
    weights = [wth11,wth12]
    segs =    [seg11,seg12]
    jacs =    [jac11,jac12]

    # create object
    obj1 = galsim.des.MultiExposureObject(images,weights,segs,jacs)

    # second obj
    img21 = numpy.ones([32,32])*211
    img22 = numpy.ones([32,32])*212
    seg21 = numpy.ones([32,32])*221
    seg22 = numpy.ones([32,32])*222
    wth21 = numpy.ones([32,32])*231
    wth22 = numpy.ones([32,32])*232
    jac21 = numpy.array([[21.1 , 21.2],[22.3 , 21.4]])
    jac22 = numpy.array([[22.1 , 22.2],[22.3 , 22.4]])

    # create lists
    images =  [img21,img22]
    weights = [wth21,wth22]
    segs =    [seg21,seg22]
    jacs =    [jac22,jac22]

    # create object
    obj2 = galsim.des.MultiExposureObject(images,weights,segs,jacs)

    # create an object list
    objlist = [obj1,obj2]

    # save objects to MEDS file
    filename_meds='test_meds.fits'
    galsim.des.write_meds(filename_meds,objlist,clobber=True)
    print 'wrote MEDS file %s ' % filename_meds

    # test functions in des_meds.py
    print 'reading %s' % filename_meds
    import meds
    m=meds.MEDS(filename_meds)
    print 'number of objects is %d' % len(m._cat)
    print 'testing if loaded images are the same as original images'

    # get the catalog
    cat=m.get_cat()

    # loop over objects and exposures
    n_obj=2
    n_cut=2
    for iobj in range(n_obj):
        for icut in range(n_cut):

            # get the images etc to compare with originals
            img=m.get_cutout( iobj, icut, type='image')
            wth=m.get_cutout( iobj, icut, type='weight')
            seg=m.get_cutout( iobj, icut, type='seg')
            jac=numpy.zeros([2,2])
            jac[0,0]=cat['dudrow'][iobj,icut]
            jac[0,1]=cat['dudcol'][iobj,icut]
            jac[1,0]=cat['dvdrow'][iobj,icut]
            jac[1,1]=cat['dvdcol'][iobj,icut]

            # compare
            numpy.testing.assert_array_equal(img,objlist[iobj].images[icut])    
            numpy.testing.assert_array_equal(wth,objlist[iobj].weights[icut])    
            numpy.testing.assert_array_equal(seg,objlist[iobj].segs[icut])    
            numpy.testing.assert_array_equal(jac,objlist[iobj].jacs[icut])    

            print 'passed obj=%d icut=%d' % (iobj,icut)

    print 'all asserts succeeded'

if __name__ == "__main__":

    test_meds()

