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
#

"""
We'll draw four test images to explore in detail the accurate generation of convolved noise fields.
This test script grew out of the Discussion on the Pull Request #366 on

https://github.com/GalSim-developers/GalSim/pull/366

We compare 'hand generated' convolved noise fields (using a COSMOS correlation function for the
underlying noise) to what GalSim thinks the theoretical correlation function should be (stored in a
correlated noise instance called `conv_cn`).

This should work for the commit up to and one beyond d85d19c5a7220496566effd466ecfdac44f7d326, but
no guarantees offered after that.

Comparing to this theoretical reference, there are four tests that generate different types of
convolved noise field (`convimage`) for comparison:

i)   the first will be direct comparison with the image created by convolving 'by hand', the
     original test on #366
ii)  the second will simply check directly-generated noise from the `conv_cn` to sanity check
     consistency
iii) the third will be like ii), but with the convimage being made slightly larger and
     then an image subset used to determine the correlation function (to test edge effects)
iv)  the fourth will be like i), but with the original COSMOS image (cosimage) being made slightly
     larger and then an image subset used to build the subsequent convimages (to test edge effects)

...for all of the above we mean subtract noise images before estimating their CF using the
subtract_mean=True keyword.

Then we look at iv) in a few extra ways:

v)   We look at iv) data using a periodicity correction with subtract_mean 'off'
vi)  We look at iv) data using a periodicity correction with subtract_mean 'on'
vii) We look at iv) data using a periodicity correction with subtract mean 'on' and a prototype
     sample variance bias correction

Finally as a controlled investigation of the sample variance bias correction, we add

viii) We look at ii) data using no periodicity correction (the data is periodic).

"""

import time
import numpy as np
import galsim

# Use a deterministic random number generator so we don't fail tests because of rare flukes
# in the random numbers.
rseed=12345

smallim_size = 16 # size of image when we test correlated noise properties using small inputs
largeim_size = 12 * smallim_size # ditto, but when we need a larger image

if __name__ == "__main__":

    t1 = time.time()
    gd = galsim.GaussianDeviate(rseed)
    dx_cosmos=0.03 # Non-unity, non-default value to be used below
    cn = galsim.getCOSMOSNoise(
        gd, '../../../examples/data/acs_I_unrot_sci_20_cf.fits', dx_cosmos=dx_cosmos)
    cn.setVariance(1000.) # Again chosen to be non-unity
    # Define a PSF with which to convolve the noise field, one WITHOUT 2-fold rotational symmetry
    # (see test_autocorrelate in test_SBProfile.py for more info as to why this is relevant)
    # Make a relatively realistic mockup of a GREAT3 target image
    lam_over_diam_cosmos = (814.e-9 / 2.4) * (180. / np.pi) * 3600. # ~lamda/D in arcsec
    lam_over_diam_ground = lam_over_diam_cosmos * 2.4 / 4. # Generic 4m at same lambda
    psf_cosmos = galsim.Convolve([
        galsim.Airy(lam_over_diam=lam_over_diam_cosmos, obscuration=0.4), galsim.Pixel(0.05)])
    psf_ground = galsim.Convolve([
        galsim.Kolmogorov(fwhm=0.8), galsim.Pixel(0.18),
        galsim.OpticalPSF(lam_over_diam=lam_over_diam_ground, coma2=0.4, defocus=-0.6)])
    psf_shera = galsim.Convolve([
        psf_ground, (galsim.Deconvolve(psf_cosmos)).createSheared(g1=0.03, g2=-0.01)])
    # Then define the convolved cosmos correlated noise model
    conv_cn = cn.copy()
    conv_cn.convolveWith(psf_shera)
    # Then draw the correlation function for this correlated noise as the reference
    refim = galsim.ImageD(smallim_size, smallim_size)
    conv_cn.draw(refim, dx=0.18)
    # Now start the tests...
    # 
    # First we generate a COSMOS noise field (cosimage), read it into an InterpolatedImage and
    # then convolve it with psf

    size_factor = .25  # scale the sizes, need size_factor * largeim_size to be an integer
    interp=galsim.Linear(tol=1.e-4) # interpolation kernel to use in making convimages
    # Number of tests
    nsum_test = 3000

    print "Calculating results for size_factor = "+str(size_factor)
    cosimage = galsim.ImageD(
        int(size_factor * largeim_size * 6), # Note 6 here since 0.18 = 6 * 0.03
        int(size_factor * largeim_size * 6)) # large image to beat down noise
    print "Unpadded underlying COSMOS noise image bounds = "+str(cosimage.bounds)
    cosimage_padded = galsim.ImageD(
        int(size_factor * largeim_size * 6) + 256, # Note 6 here since 0.18 = 6 * 0.03
        int(size_factor * largeim_size * 6) + 256) # large image to beat down noise + padding
    print "Padded underlying COSMOS noise image bounds = "+str(cosimage_padded.bounds)

    cosimage.setScale(dx_cosmos) # Use COSMOS pixel scale
    cosimage_padded.setScale(dx_cosmos) # Use COSMOS pixel scale
    cosimage.addNoise(cn)
    cosimage_padded.addNoise(cn)

    imobj = galsim.InterpolatedImage(
        cosimage, calculate_stepk=False, calculate_maxk=False, normalization='sb', dx=dx_cosmos,
        x_interpolant=interp)
    cimobj = galsim.Convolve(imobj, psf_shera)

    imobj_padded = galsim.InterpolatedImage(
        cosimage_padded, calculate_stepk=False, calculate_maxk=False,
        normalization='sb', dx=dx_cosmos, x_interpolant=interp)
    cimobj_padded = galsim.Convolve(imobj_padded, psf_shera)
 
    convimage1 = galsim.ImageD(int(largeim_size * size_factor), int(largeim_size * size_factor))
    convimage2 = galsim.ImageD(int(largeim_size * size_factor), int(largeim_size * size_factor))
    convimage4 = galsim.ImageD(int(largeim_size * size_factor), int(largeim_size * size_factor))

    print "Unpadded convolved image bounds = "+str(convimage1.bounds)
    convimage3_padded = galsim.ImageD(
        int(largeim_size * size_factor) + 32, int(largeim_size * size_factor) + 32)
    # Set the scales of convimage2 & 3 to be 0.18 so that addNoise() works correctly
    convimage2.setScale(0.18)
    convimage3_padded.setScale(0.18)
    print "Padded convolved image bounds = "+str(convimage3_padded.bounds)
    print ""

    # We draw, calculate a correlation function for the resulting field, and repeat to get an
    # average over nsum_test trials
    cimobj.draw(convimage1, dx=0.18, normalization='sb')
    cn_test1 = galsim.CorrelatedNoise(
        gd, convimage1, dx=0.18, correct_periodicity=False, subtract_mean=True,
        correct_sample_bias_prototype=False)
    testim1 = galsim.ImageD(smallim_size, smallim_size)
    cn_test1.draw(testim1, dx=0.18)
 
    convimage2.addNoise(conv_cn)  # Now we make a comparison by simply adding noise from conv_cn
    cn_test2 = galsim.CorrelatedNoise(
        gd, convimage2, dx=0.18, correct_periodicity=False, subtract_mean=True,
        correct_sample_bias_prototype=False)
    testim2 = galsim.ImageD(smallim_size, smallim_size)
    cn_test2.draw(testim2, dx=0.18)

    convimage3_padded.addNoise(conv_cn)  # Now we make a comparison by adding noise from conv_cn
    # Now only look at the subimage from convimage3, avoids edge regions which will be wrapped round
    convimage3 = convimage3_padded[convimage1.bounds]
    cn_test3 = galsim.CorrelatedNoise(
        gd, convimage3, dx=0.18, correct_periodicity=False, subtract_mean=True,
        correct_sample_bias_prototype=False)
    testim3 = galsim.ImageD(smallim_size, smallim_size)
    cn_test3.draw(testim3, dx=0.18)

    cimobj_padded.draw(convimage4, dx=0.18, normalization='sb')
    cn_test4 = galsim.CorrelatedNoise(
        gd, convimage4, dx=0.18, correct_periodicity=False, subtract_mean=True,
        correct_sample_bias_prototype=False)
    testim4 = galsim.ImageD(smallim_size, smallim_size)
    cn_test4.draw(testim4, dx=0.18)

    # Then make a testim5 which uses the noise from Case 4 but uses the periodicity correction
    cn_test5 = galsim.CorrelatedNoise(
        gd, convimage4, dx=0.18, correct_periodicity=True, subtract_mean=False)
    testim5 = galsim.ImageD(smallim_size, smallim_size)
    cn_test5.draw(testim5, dx=0.18)

    # Then make a testim6 which uses the noise from Case 4 but uses the periodicity correction and
    # turns ON the mean subtraction but doesn't use a sample bias correction
    cn_test6 = galsim.CorrelatedNoise(
        gd, convimage4, dx=0.18, correct_periodicity=True, subtract_mean=True,
        correct_sample_bias_prototype=False)
    testim6 = galsim.ImageD(smallim_size, smallim_size)
    cn_test6.draw(testim6, dx=0.18)

    # Then make a testim7 which uses the noise from Case 4 but uses the periodicity correction and
    # turns ON the mean subtraction AND uses a sample bias correction
    cn_test7 = galsim.CorrelatedNoise(
        gd, convimage4, dx=0.18, correct_periodicity=True, subtract_mean=True,
        correct_sample_bias_prototype=True)
    testim7 = galsim.ImageD(smallim_size, smallim_size)
    cn_test7.draw(testim7, dx=0.18)

    # Then make a testim8 which uses the noise from Case 2 but uses no periodicity correction and
    # turns ON the mean subtraction AND uses a sample bias correction
    cn_test8 = galsim.CorrelatedNoise(
        gd, convimage2, dx=0.18, correct_periodicity=False, subtract_mean=True,
        correct_sample_bias_prototype=True)
    testim8 = galsim.ImageD(smallim_size, smallim_size)
    cn_test8.draw(testim8, dx=0.18)

    conv1_list = [convimage1.array.copy()] # Don't forget Python reference/assignment semantics, we
                                           # zero convimage and write over it later!
    mnsq1_list = [np.mean(convimage1.array**2)]
    var1_list = [convimage1.array.var()]

    conv2_list = [convimage2.array.copy()] # Don't forget Python reference/assignment semantics, we
                                           # zero convimage and write over it later!
    mnsq2_list = [np.mean(convimage2.array**2)]
    var2_list = [convimage2.array.var()]

    conv3_list = [convimage3.array.copy()] # Don't forget Python reference/assignment semantics, we
                                           # zero convimage and write over it later!
    mnsq3_list = [np.mean(convimage3.array**2)]
    var3_list = [convimage3.array.var()]

    conv4_list = [convimage4.array.copy()] # Don't forget Python reference/assignment semantics, we
                                           # zero convimage and write over it later!
    mnsq4_list = [np.mean(convimage4.array**2)]
    var4_list = [convimage4.array.var()]

    for i in range(nsum_test - 1):
        cosimage.setZero()
        cosimage.addNoise(cn)
        cosimage_padded.setZero()
        cosimage_padded.addNoise(cn)

        imobj = galsim.InterpolatedImage(
            cosimage, calculate_stepk=False, calculate_maxk=False, normalization='sb', dx=dx_cosmos,
            x_interpolant=interp)
        cimobj = galsim.Convolve(imobj, psf_shera)

        imobj_padded = galsim.InterpolatedImage(
            cosimage_padded, calculate_stepk=False, calculate_maxk=False,
            normalization='sb', dx=dx_cosmos, x_interpolant=interp)
        cimobj_padded = galsim.Convolve(imobj_padded, psf_shera) 

        convimage1.setZero() # See above 
        convimage2.setZero() # See above
        convimage3_padded.setZero() # ditto
        convimage4.setZero() # ditto

        cimobj.draw(convimage1, dx=0.18, normalization='sb')
        conv1_list.append(convimage1.array.copy()) # See above
        mnsq1_list.append(np.mean(convimage1.array**2))
        var1_list.append(convimage1.array.var())
        cn_test1 = galsim.CorrelatedNoise(
            gd, convimage1, dx=0.18, correct_periodicity=False, subtract_mean=True,
            correct_sample_bias_prototype=False) 
        cn_test1.draw(testim1, dx=0.18, add_to_image=True)

        convimage2.addNoise(conv_cn)  # Simply adding noise from conv_cn for a comparison
        conv2_list.append(convimage2.array.copy()) # See above
        mnsq2_list.append(np.mean(convimage2.array**2))
        var2_list.append(convimage2.array.var())
        cn_test2 = galsim.CorrelatedNoise(
            gd, convimage2, dx=0.18, correct_periodicity=False, subtract_mean=True,
            correct_sample_bias_prototype=False)
        cn_test2.draw(testim2, dx=0.18, add_to_image=True)

        convimage3_padded.addNoise(conv_cn)  # Adding noise from conv_cn for a comparison
        convimage3 = convimage3_padded[convimage1.bounds]
        conv3_list.append(convimage3.array.copy()) # See above
        mnsq3_list.append(np.mean(convimage3.array**2))
        var3_list.append(convimage3.array.var())
        cn_test3 = galsim.CorrelatedNoise(
            gd, convimage3, dx=0.18, correct_periodicity=False, subtract_mean=True,
            correct_sample_bias_prototype=False)
        cn_test3.draw(testim3, dx=0.18, add_to_image=True)

        cimobj_padded.draw(convimage4, dx=0.18, normalization='sb')
        conv4_list.append(convimage4.array.copy()) # See above
        mnsq4_list.append(np.mean(convimage4.array**2))
        var4_list.append(convimage4.array.var())
        cn_test4 = galsim.CorrelatedNoise(
            gd, convimage4, dx=0.18, correct_periodicity=False, subtract_mean=True,
            correct_sample_bias_prototype=False) 
        cn_test4.draw(testim4, dx=0.18, add_to_image=True)

        # Then make a testim5 which uses the noise from Case 4 but uses the periodicity correction
        cn_test5 = galsim.CorrelatedNoise(
            gd, convimage4, dx=0.18, correct_periodicity=True, subtract_mean=False)
        cn_test5.draw(testim5, dx=0.18, add_to_image=True)

        # Then make a testim6 which uses the noise from Case 4 but uses the periodicity correction
        # and turns ON the mean subtraction but doesn't use a sample bias correction
        cn_test6 = galsim.CorrelatedNoise(
            gd, convimage4, dx=0.18, correct_periodicity=True, subtract_mean=True,
            correct_sample_bias_prototype=False)
        cn_test6.draw(testim6, dx=0.18, add_to_image=True)

        # Then make a testim7 which uses the noise from Case 4 but uses the periodicity correction
        # and turns ON the mean subtraction AND uses a sample bias correction
        cn_test7 = galsim.CorrelatedNoise(
            gd, convimage4, dx=0.18, correct_periodicity=True, subtract_mean=True,
            correct_sample_bias_prototype=True)
        cn_test7.draw(testim7, dx=0.18, add_to_image=True)

        # Then make a testim8 which uses the noise from Case 2 but uses no periodicity correction
        # and turns ON the mean subtraction AND uses a sample bias correction
        cn_test8 = galsim.CorrelatedNoise(
            gd, convimage2, dx=0.18, correct_periodicity=False, subtract_mean=True,
            correct_sample_bias_prototype=True)
        cn_test8.draw(testim8, dx=0.18, add_to_image=True)
 
        if ((i + 2) % 100 == 0): print "Completed "+str(i + 2)+"/"+str(nsum_test)+" trials"
        del imobj
        del cimobj
        del cn_test1
        del cn_test2
        del cn_test3
        del cn_test4
        del cn_test5
        del cn_test6
        del cn_test7
        del cn_test8

    mnsq1_individual = sum(mnsq1_list) / float(nsum_test)
    var1_individual = sum(var1_list) / float(nsum_test)
    mnsq2_individual = sum(mnsq2_list) / float(nsum_test)
    var2_individual = sum(var2_list) / float(nsum_test)
    mnsq3_individual = sum(mnsq3_list) / float(nsum_test)
    var3_individual = sum(var3_list) / float(nsum_test)
    mnsq4_individual = sum(mnsq4_list) / float(nsum_test)
    var4_individual = sum(var4_list) / float(nsum_test)

    testim1 /= float(nsum_test) # Take average CF of trials
    testim2 /= float(nsum_test) # Take average CF of trials
    testim3 /= float(nsum_test) # Take average CF of trials
    testim4 /= float(nsum_test) # Take average CF of trials
    testim5 /= float(nsum_test) # Take average CF of trials
    testim6 /= float(nsum_test) # Take average CF of trials
    testim7 /= float(nsum_test) # Take average CF of trials
    testim8 /= float(nsum_test) # Take average CF of trials

    refim.write('junkref.fits')
    testim1.write('junk1.fits')
    testim2.write('junk2.fits')
    testim3.write('junk3.fits')
    testim4.write('junk4.fits')
    testim5.write('junk5.fits')
    testim6.write('junk6.fits')
    testim7.write('junk7.fits')
    testim7.write('junk8.fits')

    conv1_array = np.asarray(conv1_list)
    mnsq1_all = np.mean(conv1_array**2)
    var1_all = conv1_array.var()
    conv2_array = np.asarray(conv2_list)
    mnsq2_all = np.mean(conv2_array**2)
    var2_all = conv2_array.var()
    conv3_array = np.asarray(conv3_list)
    mnsq3_all = np.mean(conv3_array**2)
    var3_all = conv3_array.var()
    conv4_array = np.asarray(conv4_list)
    mnsq4_all = np.mean(conv4_array**2)
    var4_all = conv4_array.var()

    print ""
    print "Case 1 (noise 'hand convolved'):"
    print "Mean square estimate from avg. of individual field mean squares = "+str(mnsq1_individual)
    print "Mean square estimate from all fields = "+str(mnsq1_all)
    print "Ratio of mean squares = %e" % (mnsq1_individual / mnsq1_all)
    print "Variance estimate from avg. of individual field variances = "+str(var1_individual)
    print "Variance estimate from all fields = "+str(var1_all)
    print "Ratio of variances = %e" % (var1_individual / var1_all)
    print "Zero lag CF from avg. of individual field CFs = "+str(testim1.array[8, 8])
    print "Zero lag CF in reference case = "+str(refim.array[8, 8])
    print "Ratio of zero lag CFs = %e" % (testim1.array[8, 8] / refim.array[8, 8])
    print ""
    print "Case 2 (noise generated directly from the convolved CN):"
    print "Mean square estimate from avg. of individual field mean squares = "+str(mnsq2_individual)
    print "Mean square estimate from all fields = "+str(mnsq2_all)
    print "Ratio of mean squares = %e" % (mnsq2_individual / mnsq2_all)
    print "Variance estimate from avg. of individual field variances = "+str(var2_individual)
    print "Variance estimate from all fields = "+str(var2_all)
    print "Ratio of variances = %e" % (var2_individual / var2_all)
    print "Zero lag CF from avg. of individual field CFs = "+str(testim2.array[8, 8])
    print "Zero lag CF in reference case = "+str(refim.array[8, 8])
    print "Ratio of zero lag CFs = %e" % (testim2.array[8, 8] / refim.array[8, 8])
    print ""
    print "Case 3 (noise generated directly from convolved CN, with padding to avoid adding edge "+\
        "effects):"
    print "Mean square estimate from avg. of individual field mean squares = "+str(mnsq3_individual)
    print "Mean square estimate from all fields = "+str(mnsq3_all)
    print "Ratio of mean squares = %e" % (mnsq3_individual / mnsq3_all)
    print "Variance estimate from avg. of individual field variances = "+str(var3_individual)
    print "Variance estimate from all fields = "+str(var3_all)
    print "Ratio of variances = %e" % (var3_individual / var3_all)
    print "Zero lag CF from avg. of individual field CFs = "+str(testim3.array[8, 8])
    print "Zero lag CF in reference case = "+str(refim.array[8, 8])
    print "Ratio of zero lag CFs = %e" % (testim3.array[8, 8] / refim.array[8, 8])
    print ""
    print "Case 4 (noise hand convolved, but with padding of inital image to avoid edge effects):"
    print "Mean square estimate from avg. of individual field mean squares = "+str(mnsq4_individual)
    print "Mean square estimate from all fields = "+str(mnsq4_all)
    print "Ratio of mean squares = %e" % (mnsq4_individual / mnsq4_all)
    print "Variance estimate from avg. of individual field variances = "+str(var4_individual)
    print "Variance estimate from all fields = "+str(var4_all)
    print "Ratio of variances = %e" % (var4_individual / var4_all)
    print "Zero lag CF from avg. of individual field CFs = "+str(testim4.array[8, 8])
    print "Zero lag CF in reference case = "+str(refim.array[8, 8])
    print "Ratio of zero lag CFs = %e" % (testim4.array[8, 8] / refim.array[8, 8])
    print ""
    print "Printing analysis of central 4x4 of CF from case 1, with subtract_mean=True and no corrections:"
    # Show ratios etc in central 4x4 where CF is definitely non-zero
    print 'mean diff = ',np.mean(testim1.array[4:12, 4:12] - refim.array[4:12, 4:12])
    print 'var diff = ',np.var(testim1.array[4:12, 4:12] - refim.array[4:12, 4:12])
    print 'min diff = ',np.min(testim1.array[4:12, 4:12] - refim.array[4:12, 4:12])
    print 'max diff = ',np.max(testim1.array[4:12, 4:12] - refim.array[4:12, 4:12])
    print 'mean ratio = %e' % np.mean(testim1.array[4:12, 4:12] / refim.array[4:12, 4:12])
    print 'var ratio = ',np.var(testim1.array[4:12, 4:12] / refim.array[4:12, 4:12])
    print 'min ratio = %e' % np.min(testim1.array[4:12, 4:12] / refim.array[4:12, 4:12])
    print 'max ratio = %e' % np.max(testim1.array[4:12, 4:12] / refim.array[4:12, 4:12])
    print ''
    print "Printing analysis of central 4x4 of CF from case 2, with subtract_mean=True and no corrections:"
    # Show ratios etc in central 4x4 where CF is definitely non-zero
    print 'mean diff = ',np.mean(testim2.array[4:12, 4:12] - refim.array[4:12, 4:12])
    print 'var diff = ',np.var(testim2.array[4:12, 4:12] - refim.array[4:12, 4:12])
    print 'min diff = ',np.min(testim2.array[4:12, 4:12] - refim.array[4:12, 4:12])
    print 'max diff = ',np.max(testim2.array[4:12, 4:12] - refim.array[4:12, 4:12])
    print 'mean ratio = %e' % np.mean(testim2.array[4:12, 4:12] / refim.array[4:12, 4:12])
    print 'var ratio = ',np.var(testim2.array[4:12, 4:12] / refim.array[4:12, 4:12])
    print 'min ratio = %e' % np.min(testim2.array[4:12, 4:12] / refim.array[4:12, 4:12])
    print 'max ratio = %e' % np.max(testim2.array[4:12, 4:12] / refim.array[4:12, 4:12])
    print ''
    print "Printing analysis of central 4x4 of CF from case 3, with subtract_mean=True and no corrections:"
    # Show ratios etc in central 4x4 where CF is definitely non-zero
    print 'mean diff = ',np.mean(testim3.array[4:12, 4:12] - refim.array[4:12, 4:12])
    print 'var diff = ',np.var(testim3.array[4:12, 4:12] - refim.array[4:12, 4:12])
    print 'min diff = ',np.min(testim3.array[4:12, 4:12] - refim.array[4:12, 4:12])
    print 'max diff = ',np.max(testim3.array[4:12, 4:12] - refim.array[4:12, 4:12])
    print 'mean ratio = %e' % np.mean(testim3.array[4:12, 4:12] / refim.array[4:12, 4:12])
    print 'var ratio = ',np.var(testim3.array[4:12, 4:12] / refim.array[4:12, 4:12])
    print 'min ratio = %e' % np.min(testim3.array[4:12, 4:12] / refim.array[4:12, 4:12])
    print 'max ratio = %e' % np.max(testim3.array[4:12, 4:12] / refim.array[4:12, 4:12])
    print ''
    print "Printing analysis of central 4x4 of CF from case 4, with subtract_mean=True and no corrections:"
    # Show ratios etc in central 4x4 where CF is definitely non-zero
    print 'mean diff = ',np.mean(testim4.array[4:12, 4:12] - refim.array[4:12, 4:12])
    print 'var diff = ',np.var(testim4.array[4:12, 4:12] - refim.array[4:12, 4:12])
    print 'min diff = ',np.min(testim4.array[4:12, 4:12] - refim.array[4:12, 4:12])
    print 'max diff = ',np.max(testim4.array[4:12, 4:12] - refim.array[4:12, 4:12])
    print 'mean ratio = %e' % np.mean(testim4.array[4:12, 4:12] / refim.array[4:12, 4:12])
    print 'var ratio = ',np.var(testim4.array[4:12, 4:12] / refim.array[4:12, 4:12])
    print 'min ratio = %e' % np.min(testim4.array[4:12, 4:12] / refim.array[4:12, 4:12])
    print 'max ratio = %e' % np.max(testim4.array[4:12, 4:12] / refim.array[4:12, 4:12])
    print ''
    print "Printing analysis of central 4x4 of CF from case 4 using case 3 as the reference:"
    # Show ratios etc in central 4x4 where CF is definitely non-zero
    print 'mean diff = ',np.mean(testim4.array[4:12, 4:12] - testim3.array[4:12, 4:12])
    print 'var diff = ',np.var(testim4.array[4:12, 4:12] - testim3.array[4:12, 4:12])
    print 'min diff = ',np.min(testim4.array[4:12, 4:12] - testim3.array[4:12, 4:12])
    print 'max diff = ',np.max(testim4.array[4:12, 4:12] - testim3.array[4:12, 4:12])
    print 'mean ratio = %e' % np.mean(testim4.array[4:12, 4:12] / testim3.array[4:12, 4:12])
    print 'var ratio = ',np.var(testim4.array[4:12, 4:12] / testim3.array[4:12, 4:12])
    print 'min ratio = %e' % np.min(testim4.array[4:12, 4:12] / testim3.array[4:12, 4:12])
    print 'max ratio = %e' % np.max(testim4.array[4:12, 4:12] / testim3.array[4:12, 4:12])
    print ''
    print 'Printing analysis of central 4x4 of CF from case 4 using periodicity correction '+\
        'with subtract_mean=False (Case 5):'
    # Show ratios etc in central 4x4 where CF is definitely non-zero
    print 'mean diff = ',np.mean(testim5.array[4:12, 4:12] - refim.array[4:12, 4:12])
    print 'var diff = ',np.var(testim5.array[4:12, 4:12] - refim.array[4:12, 4:12])
    print 'min diff = ',np.min(testim5.array[4:12, 4:12] - refim.array[4:12, 4:12])
    print 'max diff = ',np.max(testim5.array[4:12, 4:12] - refim.array[4:12, 4:12])
    print 'mean ratio = %e' % np.mean(testim5.array[4:12, 4:12] / refim.array[4:12, 4:12])
    print 'var ratio = ',np.var(testim5.array[4:12, 4:12] / refim.array[4:12, 4:12])
    print 'min ratio = %e' % np.min(testim5.array[4:12, 4:12] / refim.array[4:12, 4:12])
    print 'max ratio = %e' % np.max(testim5.array[4:12, 4:12] / refim.array[4:12, 4:12])
    print ''
    print 'Printing analysis of central 4x4 of CF from case 4 using periodicity correction '+\
        'with subtract_mean=True but no sample bias correction (Case 6):'
    # Show ratios etc in central 4x4 where CF is definitely non-zero
    print 'mean diff = ',np.mean(testim6.array[4:12, 4:12] - refim.array[4:12, 4:12])
    print 'var diff = ',np.var(testim6.array[4:12, 4:12] - refim.array[4:12, 4:12])
    print 'min diff = ',np.min(testim6.array[4:12, 4:12] - refim.array[4:12, 4:12])
    print 'max diff = ',np.max(testim6.array[4:12, 4:12] - refim.array[4:12, 4:12])
    print 'mean ratio = %e' % np.mean(testim6.array[4:12, 4:12] / refim.array[4:12, 4:12])
    print 'var ratio = ',np.var(testim6.array[4:12, 4:12] / refim.array[4:12, 4:12])
    print 'min ratio = %e' % np.min(testim6.array[4:12, 4:12] / refim.array[4:12, 4:12])
    print 'max ratio = %e' % np.max(testim6.array[4:12, 4:12] / refim.array[4:12, 4:12])
    print ''
    print 'Printing analysis of central 4x4 of CF from case 4 using periodicity correction '+\
        'with subtract_mean=True but with a sample bias correction (Case 7):'
    # Show ratios etc in central 4x4 where CF is definitely non-zero
    print 'mean diff = ',np.mean(testim7.array[4:12, 4:12] - refim.array[4:12, 4:12])
    print 'var diff = ',np.var(testim7.array[4:12, 4:12] - refim.array[4:12, 4:12])
    print 'min diff = ',np.min(testim7.array[4:12, 4:12] - refim.array[4:12, 4:12])
    print 'max diff = ',np.max(testim7.array[4:12, 4:12] - refim.array[4:12, 4:12])
    print 'mean ratio = %e' % np.mean(testim7.array[4:12, 4:12] / refim.array[4:12, 4:12])
    print 'var ratio = ',np.var(testim7.array[4:12, 4:12] / refim.array[4:12, 4:12])
    print 'min ratio = %e' % np.min(testim7.array[4:12, 4:12] / refim.array[4:12, 4:12])
    print 'max ratio = %e' % np.max(testim7.array[4:12, 4:12] / refim.array[4:12, 4:12])
    print ''
    print 'Printing analysis of central 4x4 of CF from case 2 using no periodicity correction '+\
        'with subtract_mean=True but with a sample bias correction (Case 8):'
    # Show ratios etc in central 4x4 where CF is definitely non-zero
    print 'mean diff = ',np.mean(testim8.array[4:12, 4:12] - refim.array[4:12, 4:12])
    print 'var diff = ',np.var(testim8.array[4:12, 4:12] - refim.array[4:12, 4:12])
    print 'min diff = ',np.min(testim8.array[4:12, 4:12] - refim.array[4:12, 4:12])
    print 'max diff = ',np.max(testim8.array[4:12, 4:12] - refim.array[4:12, 4:12])
    print 'mean ratio = %e' % np.mean(testim8.array[4:12, 4:12] / refim.array[4:12, 4:12])
    print 'var ratio = ',np.var(testim8.array[4:12, 4:12] / refim.array[4:12, 4:12])
    print 'min ratio = %e' % np.min(testim8.array[4:12, 4:12] / refim.array[4:12, 4:12])
    print 'max ratio = %e' % np.max(testim8.array[4:12, 4:12] / refim.array[4:12, 4:12])
    print ''
