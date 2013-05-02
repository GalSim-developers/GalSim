import sys
import logging
import galsim

"""
A test of the behavior of shear, magnification, shift, rotation, etc for different interpolants and gsparams when applied to RealGalaxy objects.
"""
# Interpolants to test
interpolant_list = ['nearest', 'sinc', 'linear', 'cubic', 'quintic', 
                    'lanczos3', 'lanczos4', 'lanczos5', 'lanczos7']

# Noise_pad options to test
noise_padding_list = ['False']

# Padding options to test
padding_list = range(2,6,2)

# Range of rotation angles to test
angle_list = galsim.degrees*range(0,180,15)

# Range of shears to test (must be same length as magnification list)
shear_list = []

# Range of magnifications to test (must be same length as shear list)
magnification_list = []

# Ground-image parameters
atmos_fwhm = 2.1
atmos_e = 0.13         # 
atmos_beta = 0.81      # radians
opt_defocus=0.53       # wavelengths
opt_a1=-0.29           # wavelengths
opt_a2=0.12            # wavelengths
opt_c1=0.64            # wavelengths
opt_c2=-0.33           # wavelengths
opt_obscuration=0.3    # linear scale size of secondary mirror obscuration
lam = 800              # nm    NB: don't use lambda - that's a reserved word.
tel_diam = 4.          # meters 
pixel_scale = 0.23     # arcsec / pixel

atmos = galsim.Kolmogorov(fwhm=atmos_fwhm)
atmos.applyShear(e=atmos_e, beta=atmos_beta*galsim.radians)
    lam_over_diam = 
optics = galsim.OpticalPSF(lam * 1.e-9 / tel_diam * 206265,  #lambda/diameter in arcsec 
                           defocus = opt_defocus,
                           coma1 = opt_c1, coma2 = opt_c2,
                           astig1 = opt_a1, astig2 = opt_a2,
                           obscuration = opt_obscuration)
psf = galsim.Convolve([atmos, optics])

# Space-image parameters
dx_space = 
imsize_space =
psf_space = 



def run_tests(config, shear, dilation, rotation, shift):

    # Set up a config dict to replicate the GSObject spec above
    config = {}

    config['gal'] = {
        "type" : "Sersic",
        "n" : galn,
        "half_light_radius" : galhlr,
        "ellip" : {
            "type" : "G1G2",
            "g1" : g1gal,
            "g2" : g2gal
        }
    }

    config['psf'] = {
        "type" : "Moffat",
        "beta" : psfbeta,
        "fwhm" : psffwhm, 
        "ellip" : {
            "type" : "G1G2",
            "g1" : g1psf,
            "g2" : g2psf
        }
    }

    config['image'] = {
        'size' : imsize,
        'pixel_scale' : dx,
        'random_seed' : rseed,
        'wmult' : wmult
    }

    # Use an automatically-determined N core run setting
    print "Starting tests using config file with N_PHOTONS = "+str(np)
    res8 = galsim.utilities.compare_dft_vs_photon_config(
        config, n_photons_per_trial=np, nproc=-1, logger=logger, abs_tol_ellip=tol_ellip,
        abs_tol_size=tol_size)
    print res8
    return

def main():
    for interpolant in interpolant_list:
        for padding_type in noise_padding_list:
            for padding in padding_list:
                run_dft_tests(plain_galaxy)
                run_dftshoot_tests(plain_original_galaxy)
                for angle in angle_list:
                    run_dft_tests(rotated_galaxy)
                    run_dft_shoot_tests(rotated_original_galaxy)
                for (shear, mag) in zip(shear_list, magnification_list):
                    run_dft_tests(sheared_and_magnified_galaxy)
                    run_dft_shoot_tests(sheared_and_magnified_original_galaxy)
                
        
        

    config = {}    
    config['gal'] = 
    
gal :
    type : RealGalaxy
    id : 
        # We select randomly among these 5 IDs.
        type : List
        items : [ 106416, 106731, 108402, 116045, 116448 ]
        index : { type : Random }

    dilation :
        # We draw the sizes from a power law distribution.
        # You can use any arbitrary function here, along with a min and max value,
        # and values will be drawn within this probability distribution.
        # The function should be given in terms of the variable x, since internally
        # it is parsed with eval('lambda x : ' + function).
        type : RandomDistribution
        function : x**-3.5
        x_min : 1
        x_max : 5

    shear : 
        # We again use PowerSpectrumShear, which is set up below using input : power_spectrum.
        type : PowerSpectrumShear

    magnification :
        # We use PowerSpectrumMagnification for this, which is set up below using input : 
        # power_spectrum.
        type : PowerSpectrumMagnification

    rotation :
        type : Random

    

if __name__ == "__main__":
    main()


