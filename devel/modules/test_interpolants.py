import sys
import logging
import galsim

"""
A test of the behavior of shear, magnification, shift, rotation, etc for different interpolants and gsparams when applied to RealGalaxy objects.
"""
# --- THINGS WE ARE TESTING  ---
# Interpolants
interpolant_list = ['nearest', 'sinc', 'linear', 'cubic', 'quintic', 
                    'lanczos3', 'lanczos4', 'lanczos5', 'lanczos7']
# Noise_pad options
noise_padding_list = ['False']
# Padding options
padding_list = range(2,6,2)
# Range of rotation angles
angle_list = galsim.degrees*range(0,180,15)
# Range of shears (must be same length as magnification list)
shear_list = []
# Range of magnifications (must be same length as shear list)
magnification_list = []

# --- IMPORTANT BUT NOT TESTED PARAMETERS ---
# Galaxy indices
galaxy_indices = []

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

# Space-image parameters
dx_space = 
imsize_space =
psf_space = 

def get_config():
# A function to return two config dictionaries with the basic PSF info that we will reuse for all
# tests.
    space_config = {}
    ground_config = {}

    space_config['psf'] = { "type" : "Moffat", "beta" : psfbeta, "fwhm" : psffwhm, 
        "ellip" : { "type" : "G1G2", "g1" : g1psf, "g2" : g2psf } }
        
    config['image'] = {
        'size' : imsize,
        'pixel_scale' : dx,
        'random_seed' : rseed,
        'wmult' : wmult
    }

    ground_config['psf'] = { 
        'type' : 'Convolution', 
        'items' : [ {'type' : 'Kolmogorov', 'fwhm' : atmos_fwhm, 
                     'ellip' : { 'type' : 'EBeta', 'e' : atmos_e, 'beta' : atmos_beta} },
                    {'type' : 'OpticalPSF', 'lam_over_diam' : lam*1.E-9/tel_diam*206265,
                     'defocus' : opt_defocus, 'astig1' : opt_a1, 'astig2' : opt_a2,
                     'coma1' : opt_c1, 'coma2' : opt_c2, 'obscuration' : opt_obscuration } ] 
    }
                
    return [space_config, ground_config]

class InterpolationData:
# Quick container class for passing around data from these tests.  
    def __init__(config, g1_obs, g2_obs, sigma_obs, err_g1_obs, err_g2_obs, err_sigma_obs):
        self.g1_obs = g1_obs
        self.g2_obs = g2_obs
        self.sigma_obs = sigma_obs
        self.err_g1_obs = err_g1_obs
        self.err_g2_obs = err_g2_obs
        self.err_sigma_obs = err_sigma_obs

def test_realgalaxy(config, shear, dilation, rotation, shift):
    # Use an automatically-determined N core run setting
    print "Starting tests using config file with N_PHOTONS = "+str(np)
    res8 = galsim.utilities.compare_dft_vs_photon_config(
        config, n_photons_per_trial=np, nproc=-1, logger=logger, abs_tol_ellip=tol_ellip,
        abs_tol_size=tol_size)
    print res8
    return

def test_realgalaxyoriginal(config, shear, dilation, rotation, shift):
# Test values for 
    # Use an automatically-determined N core run setting
    print "Starting tests using config file with N_PHOTONS = "+str(np)
    result = galsim.utilities.compare_dft_vs_photon_config(
        config, n_photons_per_trial=np, nproc=-1, logger=logger, abs_tol_ellip=tol_ellip,
        abs_tol_size=tol_size)
    return InterpolationData(config, g1_obs, g2_obs, sigma_obs, err_g1_obs, err_g2_obs, err_sigma_obs)



def main():
    # Define the config dictionaries we will use for all the following tests
    config_list = get_config()
    
    # Now, run through the various things we need to test in loops.
    # Right now, test rotation angles separately from shear and magnification
    # (but we can do that nested later if need be - probably with fewer tested angles).
    for base_config in config_list:                     # Ground and space
        base_answer = test_realgalaxy(base_galaxy)
        base_answer_dft, base_answer_shoot = test_realgalaxyoriginal(base_galaxy_original)
        for angle in angle_list:                        # Possible rotation angles
            for interpolant in interpolant_list:        # Possible interpolants
                for padding_type in noise_padding_list: # Noise pad or not
                    for padding in padding_list:        # Amount of padding
                        print_results(base_answer, test_realgalaxy(rotated_galaxy))
                        print_results_original(base_answer_dft, base_answer_shoot, 
                                               test_realgalaxyoriginal(rotated_original_galaxy))
        for (shear, mag) in zip(shear_list, magnification_list): # Shear and magnification
            for interpolant in interpolant_list:                 # Possible interpolants
                for padding_type in noise_padding_list:          # Noise pad or not
                    for padding in padding_list:                 # Amount of padding
                        print_results(base_answer, test_realgalaxy(rotated_galaxy))
                        print_results_original(base_answer_dft, base_answer_shoot, 
                                               test_realgalaxyoriginal(rotated_original_galaxy))
                
        
        

    
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

