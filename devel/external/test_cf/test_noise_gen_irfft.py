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
"""@file test_noise_gen_irfft.py

This is a simple test module used to make plots that led to a deeper understanding of what was
going wrong in Barney's attempts to make the generation of correlated noise more efficient via the
use of Hermitian symmetry and numpy.fft.irfftn.

See https://github.com/GalSim-developers/GalSim/issues/563

The backstory is that modifying galsim/correlatednoise.py to use only the half of the (root) power
spectrum, multiplying that by a complex random number z = (x + iy)/sqrt(2) where x,y~N(0,1) and then
using irfft2 to inverse transform to get the realization of the noise field with the given power
spectrum... wasn't working properly.  It was close, but one of the more sensitive "end to end" tests
in the tests/test_correlatednoise.py suite was failing (test_convolve_cosmos) when the precision
was ramped up.

More details are posted on the issue linked above.
"""
import sys
import numpy as np

# Define sizes of arrays, input parameters
sigma = 0.17           # our noise field is going to have a Gaussian CF & PS for simplicity
nu = 22                # number of array elements
u = np.fft.fftfreq(nu) # get the u0, u1, u2 etc. frequency values
nsamples = 50000       # number of realizations to average over for 1D tests
ps = np.exp(-2. * np.pi**2 * sigma**2 * u**2) # using result for FT of a Gaussian being another one
                                              # (note scale factors in this might be wrong, done
                                              # from memory, but not crucial for demonstration)

# 1D plots first
print "Generating 1D plots..."
psests = []
psests_r = []
for i in range(nsamples):

    realization = np.fft.ifft((np.random.randn(nu) + 1j * np.random.randn(nu)) * np.sqrt(ps)).real
    realization_irfft = np.fft.irfft(
        (np.random.randn(nu//2+1) + 1j * np.random.randn(nu//2+1)) * np.sqrt(.5 * ps[:nu//2+1]), nu)
    psests.append(np.abs(np.fft.fft(realization))**2)
    psests_r.append(np.abs(np.fft.fft(realization_irfft))**2)

psest = np.mean(np.array(psests), axis=0)
psest_r = np.mean(np.array(psests_r), axis=0)

cfest = np.fft.irfft(psest[:nu//2+1])
cfest_r = np.fft.irfft(psest_r[:nu//2+1])
cf = np.fft.irfft(ps[:nu//2+1])


import matplotlib.pyplot as plt
# Plot the power spectra

plt.clf()
plt.plot(psest, label="fft")
plt.plot(psest_r, label="rfft")
plt.plot(ps, label="reference", lw=2, color="k", ls=":")
plt.ylabel("Power spectrum")
plt.legend()
plt.savefig("ps_irfft.png")

plt.clf()
plt.plot(cfest, label="fft")
plt.plot(cfest_r, label="rfft")
plt.plot(cf, label="reference", lw=2, color="k", ls=":")
plt.ylabel("Correlation function")
plt.legend()
plt.savefig("cf_irfft.png")

# Then do 2D tests
nsamplesxy = 1000000

# NEWFIX - this is an attempt to reftify the code prior to this point in 2D, which seems to include
# quite a bit of muddled thinking!
# See: https://github.com/GalSim-developers/GalSim/issues/563#issuecomment-47344100
print "Generating 2D plots with Gary's fix..."
rt2 = np.sqrt(2.)
irt2 = 1. / rt2
for nu in (21, 22): # number of array elements, odd first then even

    u = np.fft.fftfreq(nu) # get the u0, u1, u2 etc. frequency values
    print "Running case with nu = "+str(nu)
    # Set up the 2D arrays and PS
    ux, uy = np.meshgrid(u, u)
    psxy = np.exp(-2. * np.pi**2 * sigma**2 * (ux**2 + uy**2))
    cf = np.fft.irfft2(psxy[:, :ux.shape[1] // 2 + 1], s=ux.shape)
    halfpsxy = psxy[:, :ux.shape[1]//2+1].copy()
    # Result storage array
    psxyest_r_fixed = np.zeros_like(ux)
    for i in range(nsamplesxy):

        randvec_real = np.random.randn(ux.shape[0], ux.shape[1]//2 + 1)
        randvec_imag = np.random.randn(ux.shape[0], ux.shape[1]//2 + 1)
        randvec = (randvec_real + 1j * randvec_imag) * irt2
        # Then do Gary's suggested fix
        # First the part of the fix that needs to happen for both odd and even sized arrays
        randvec[ux.shape[0]-1: ux.shape[0]/2:-1, 0] = np.conj(randvec[1: (ux.shape[0]+1)/2, 0])
        randvec[0, 0] = rt2 * randvec[0, 0].real
        # Then make the changes required for even sized arrays
        if ux.shape[1] % 2 == 0:
            randvec[-1: ux.shape[0]/2:-1, ux.shape[0]/2] = np.conj(
                randvec[1: (ux.shape[0]+1)/2, ux.shape[1]/2])
            randvec[0, ux.shape[1]/2] = rt2 * randvec[0, ux.shape[1]/2].real
        if ux.shape[0] % 2 == 0:
            randvec[ux.shape[0]/2, 0] = rt2 * randvec[ux.shape[0]/2, 0].real
            if ux.shape[1] % 2 == 0:
                randvec[ux.shape[0]/2, ux.shape[1]/2] = \
                    rt2 * randvec[ux.shape[0]/2, ux.shape[1]/2].real
        realization_irfft = np.fft.irfft2(randvec * np.sqrt(halfpsxy), s=ux.shape)
        psxyest_r_fixed += np.abs(np.fft.fft2(realization_irfft))**2
        if i % 1000 == 0:
            sys.stdout.write(
                "Completed: %d%%      %s" % (
                    int(np.round(100. * float(i) / float(nsamplesxy))), "\r"))
            sys.stdout.flush()

    psxyest_r_fixed /= float(nsamplesxy)
    cfest_r_fixed = np.fft.irfft2(psxyest_r_fixed[:, :ux.shape[1]//2 + 1], s=ux.shape)

    # Make some plots!
    
    # PS ref, fixed and difference
    plt.clf()
    plt.pcolor(psxy)
    plt.colorbar()
    plt.xlim(0, ux.shape[1])
    plt.ylim(0, ux.shape[0])
    plt.savefig("psxy_ref_nu"+str(nu)+"_N"+str(nsamplesxy)+".png")
    plt.clf()
    plt.pcolor(psxyest_r_fixed)
    plt.colorbar()
    plt.xlim(0, ux.shape[1])
    plt.ylim(0, ux.shape[0])
    plt.savefig("psxy_rfft_gary_fixed_nu"+str(nu)+"_N"+str(nsamplesxy)+".png")
    plt.clf()
    plt.pcolor(psxyest_r_fixed - psxy)
    plt.colorbar()
    plt.xlim(0, ux.shape[1])
    plt.ylim(0, ux.shape[0])
    plt.savefig("psxy_rfft_gary_fixed-ref_nu"+str(nu)+"_N"+str(nsamplesxy)+".png")
    
    # CF ref, fixed and difference
    plt.clf()
    plt.pcolor(
        galsim.utilities.roll2d(cf, (ux.shape[0]/2, ux.shape[1]/2)))
    plt.colorbar()
    plt.xlim(0, ux.shape[1])
    plt.ylim(0, ux.shape[0])
    plt.savefig("cfxy_ref_nu"+str(nu)+"_N"+str(nsamplesxy)+".png")
    plt.clf()
    plt.pcolor(
        galsim.utilities.roll2d(cfest_r_fixed, (ux.shape[0]/2, ux.shape[1]/2)))
    plt.colorbar()
    plt.xlim(0, ux.shape[1])
    plt.ylim(0, ux.shape[0])
    plt.savefig("cfxy_rfft_gary_fixed_nu"+str(nu)+"_N"+str(nsamplesxy)+".png")
    plt.clf()
    plt.pcolor(
        galsim.utilities.roll2d(cfest_r_fixed - cf, (ux.shape[0]/2, ux.shape[1]/2)))
    plt.colorbar()
    plt.xlim(0, ux.shape[1])
    plt.ylim(0, ux.shape[0])
    plt.savefig("cfxy_rfft_gary_fixed-cf_nu"+str(nu)+"_N"+str(nsamplesxy)+".png")

print "Generating 2D plots, odd/even sized, with Gary's fix..."
rt2 = np.sqrt(2.)
irt2 = 1. / rt2
for nu in (21, 22): # number of array elements, odd first then even

    u1 = np.fft.fftfreq(nu) # get the u0, u1, u2 etc. frequency values
    u2 = np.fft.fftfreq(nu + 1) # get the u0, u1, u2 etc. frequency values
    print "Running case with nu = "+str(nu)
    # Set up the 2D arrays and PS
    ux, uy = np.meshgrid(u1, u2)
    psxy = np.exp(-2. * np.pi**2 * sigma**2 * (ux**2 + uy**2))
    cf = np.fft.irfft2(psxy[:, :ux.shape[1] // 2 + 1], s=ux.shape)
    halfpsxy = psxy[:, :ux.shape[1]//2+1].copy()
    # Result storage array
    psxyest_r_fixed = np.zeros_like(ux)
    for i in range(nsamplesxy):

        randvec_real = np.random.randn(ux.shape[0], ux.shape[1]//2 + 1)
        randvec_imag = np.random.randn(ux.shape[0], ux.shape[1]//2 + 1)
        randvec = (randvec_real + 1j * randvec_imag) * irt2
        # Then do Gary's suggested fix
        # First the part of the fix that needs to happen for both odd and even sized arrays
        randvec[ux.shape[0]-1: ux.shape[0]/2:-1, 0] = np.conj(randvec[1: (ux.shape[0]+1)/2, 0])
        randvec[0, 0] = rt2 * randvec[0, 0].real
        # Then make the changes required for even sized arrays
        if ux.shape[1] % 2 == 0:
            randvec[-1: ux.shape[0]/2:-1, ux.shape[1]/2] = np.conj(
                randvec[1: (ux.shape[0]+1)/2, ux.shape[1]/2])
            randvec[0, ux.shape[1]/2] = rt2 * randvec[0, ux.shape[1]/2].real
        if ux.shape[0] % 2 == 0:
            randvec[ux.shape[0]/2, 0] = rt2 * randvec[ux.shape[0]/2, 0].real
            if ux.shape[1] % 2 == 0:
                randvec[ux.shape[0]/2, ux.shape[1]/2] = \
                    rt2 * randvec[ux.shape[0]/2, ux.shape[1]/2].real
        realization_irfft = np.fft.irfft2(randvec * np.sqrt(halfpsxy), s=ux.shape)
        psxyest_r_fixed += np.abs(np.fft.fft2(realization_irfft))**2
        if i % 1000 == 0:
            sys.stdout.write(
                "Completed: %d%%      %s" % (
                    int(np.round(100. * float(i) / float(nsamplesxy))), "\r"))
            sys.stdout.flush()

    psxyest_r_fixed /= float(nsamplesxy)
    cfest_r_fixed = np.fft.irfft2(psxyest_r_fixed[:, :ux.shape[1]//2 + 1], s=ux.shape)

    # Make some plots!
    import galsim
    # PS ref, fixed and difference
    plt.clf()
    plt.pcolor(psxy)
    plt.colorbar()
    plt.xlim(0, ux.shape[1])
    plt.ylim(0, ux.shape[0])
    plt.savefig("psxy_ref_nu1"+str(len(u1))+"_nu2"+str(len(u2))+"_N"+str(nsamplesxy)+".png")
    plt.clf()
    plt.pcolor(psxyest_r_fixed)
    plt.colorbar()
    plt.xlim(0, ux.shape[1])
    plt.ylim(0, ux.shape[0])
    plt.savefig(
        "psxy_rfft_gary_fixed_nu1"+str(len(u1))+"_nu2"+str(len(u2))+"_N"+str(nsamplesxy)+".png")
    plt.clf()
    plt.pcolor(psxyest_r_fixed - psxy)
    plt.colorbar()
    plt.xlim(0, ux.shape[1])
    plt.ylim(0, ux.shape[0])
    plt.savefig(
        "psxy_rfft_gary_fixed-ref_nu1"+str(len(u1))+"_nu2"+str(len(u2))+"_N"+str(nsamplesxy)+".png")
    
    # CF ref, fixed and difference
    plt.clf()
    plt.pcolor(
        galsim.utilities.roll2d(cf, (ux.shape[0]/2, ux.shape[1]/2)))
    plt.colorbar()
    plt.xlim(0, ux.shape[1])
    plt.ylim(0, ux.shape[0])
    plt.savefig(
        "cfxy_ref_nu1"+str(len(u1))+"_nu2"+str(len(u2))+"_N"+str(nsamplesxy)+".png")
    plt.clf()
    plt.pcolor(
        galsim.utilities.roll2d(cfest_r_fixed, (ux.shape[0]/2, ux.shape[1]/2)))
    plt.colorbar()
    plt.xlim(0, ux.shape[1])
    plt.ylim(0, ux.shape[0])
    plt.savefig(
        "cfxy_rfft_gary_fixed_nu1"+str(len(u1))+"_nu2"+str(len(u2))+"_N"+str(nsamplesxy)+".png")
    plt.clf()
    plt.pcolor(
        galsim.utilities.roll2d(cfest_r_fixed - cf, (ux.shape[0]/2, ux.shape[1]/2)))
    plt.colorbar()
    plt.xlim(0, ux.shape[1])
    plt.ylim(0, ux.shape[0])
    plt.savefig(
        "cfxy_rfft_gary_fixed-cf_nu1"+str(len(u1))+"_nu2"+str(len(u2))+"_N"+str(nsamplesxy)+".png")

    


