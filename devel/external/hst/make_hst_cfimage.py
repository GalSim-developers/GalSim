import os
import cPickle
import numpy as np
import galsim

NOISEIMFILE="acs_I_unrot_sci_20_noisearrays.pkl"
CFIMFILE="acs_I_unrot_sci_20_cf.fits"
CFPLOTFILE="acs_I_unrot_sci_20_cf.png"
CFLOGPLOTFILE="acs_I_unrot_sci_20_log10cf.png"
NPIX=81
PIXSCALE_ACS_ARCSEC=0.03

if not os.path.isfile(CFIMFILE):
    # Read in the pickled images
    noiseims = cPickle.load(open(NOISEIMFILE, 'rb'))
    # Loop through the images and sum the correlation functions
    hst_ncf = None
    for noiseim in noiseims:
        noiseim = noiseim.astype(np.float64)
        # Subtract off the mean for each field (bg subtraction never perfect)
        noiseim -= noiseim.mean()
        if hst_ncf is None:
            # Initialize the HST noise correlation function using the first image
            hst_ncf = galsim.ImageCorrFunc(galsim.ImageViewD(noiseim))
        else:
            hst_ncf += galsim.ImageCorrFunc(galsim.ImageViewD(noiseim))
    # Draw and plot an output image of the resulting correlation function
    cfimage = galsim.ImageD(NPIX, NPIX)
    hst_ncf.draw(cfimage, dx=1.)
    # Save this to the output filename specified in the script header
    cfimage.write(CFIMFILE)
else:
    cfimage = galsim.fits.read(CFIMFILE)

# Then make some nice plots
import matplotlib.pyplot as plt
plt.clf()
plt.pcolor(cfimage.array, vmin=0.)
plt.axis((0, NPIX, 0, NPIX))
plt.colorbar()
plt.set_cmap('hot')
plt.title(r'COSMOS F814W-unrotated-sci noise correlation function')
plt.savefig(CFPLOTFILE)
plt.show()
plt.clf()
plt.pcolor(np.log10(cfimage.array + 5.e-6))
plt.axis((0, NPIX, 0, NPIX))
plt.colorbar()
plt.set_cmap('hot')
plt.title('log10 COSMOS F814W-unrotated-sci noise correlation function')
plt.savefig(CFLOGPLOTFILE)
plt.show()

        
