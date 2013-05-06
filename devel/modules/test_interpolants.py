"""
A test of the behavior of shear, magnification, shift, rotation, etc for different interpolants and gsparams when applied to RealGalaxy objects.
"""

import sys
import logging
import galsim
import copy

# --- THINGS WE ARE TESTING  ---
# Interpolants
interpolant_list = ['nearest', 'sinc', 'linear', 'cubic', 'quintic', 
                    'lanczos3', 'lanczos4', 'lanczos5', 'lanczos7']
# Padding options
padding_list = range(2,6,2)
# Range of rotation angles
angle_list = range(0,180,15)
# Range of shears and magnifications to use
# This is currently the 4 largest and 4 smallest |g|s generated from a run of examples/demo11.py,
# rounded to 1-3 sig figs, plus the 4 largest blown up by a factor of 30 to really strain the 
# algorithm.
shear_and_magnification_list = [(0.0003, 0.0003, 1.01), (0.0005, 0.0001, 0.96), 
                                (-0.0004, 0.0009, 1.03), (0.001, -0.0003, 0.996), 
                                (0.015, -0.010, 1.02), (-0.016, -0.009, 1.01), 
                                (0.016, -0.011, 1.02), (0.020, 0.003, 0.97), 
                                ( 0.46, -0.31,  1.6), ( -0.49, -0.26,  1.4), 
                                ( 0.47, -0.33,  1.5), ( 0.61, 0.095,  0.16)]


# --- IMPORTANT BUT NOT TESTED PARAMETERS ---

# Galaxy indices
first_index = 10000
nitems = 1

# Ground-image parameters
atmos_fwhm = 2.1
atmos_e = 0.13             
atmos_beta = 0.81         # radians
opt_defocus=0.53          # wavelengths
opt_a1=-0.29              # wavelengths
opt_a2=0.12               # wavelengths
opt_c1=0.64               # wavelengths
opt_c2=-0.33              # wavelengths
opt_obscuration=0.3       # linear scale size of secondary mirror obscuration
lam = 800                 # nm    NB: don't use lambda - that's a reserved word.
tel_diam = 4.             # meters 
pixel_scale_ground = 0.23 # arcsec / pixel
imsize_ground =

# Space-image parameters
pixel_scale_space = 
imsize_space =
beta_space = 
fwhm_space = 
g1psf_space =
g2psf_space =

# --- Machine-specific parameters ---
nproc = 4

# --- Helper functions to run the main part of the code ---
def get_config():
# A function to return two config dictionaries with the basic PSF info that we will reuse for all
# tests.
    space_config = {}
    ground_config = {}

    space_config['psf'] = { "type" : "Moffat", "beta" : beta_space, "fwhm" : fwhm_space, 
        "ellip" : { "type" : "G1G2", "g1" : g1psf_space, "g2" : g2psf_space } }
        
    ground_config['psf'] = { 
        'type' : 'Convolution', 
        'items' : [ {'type' : 'Kolmogorov', 'fwhm' : atmos_fwhm, 
                     'ellip' : { 'type' : 'EBeta', 'e' : atmos_e, 'beta' : atmos_beta} },
                    {'type' : 'OpticalPSF', 'lam_over_diam' : lam*1.E-9/tel_diam*206265,
                     'defocus' : opt_defocus, 'astig1' : opt_a1, 'astig2' : opt_a2,
                     'coma1' : opt_c1, 'coma2' : opt_c2, 'obscuration' : opt_obscuration } ] 
    }
                
    space_config['image'] = {
        'type' : 'Single',
        'size' : imsize_space,
        'pixel_scale' : pixel_scale_space,
        'nproc' : nproc
    }

    ground_config['image'] = {
        'type' : 'Single',
        'size' : imsize_ground,
        'pixel_scale' : pixel_scale_ground,
        'nproc' : nproc
    }
    
    galaxy_config = { 'type': 'RealGalaxy', 
                      'index': { 'type': 'Sequence', 'first': first_index, 'nitems': nitems } }
    catalog_config = {'real_catalog' : { 'dir' : '../examples/data', 
        'file_name' :  'real_galaxy_catalog_example.fits', 'preload' : True} }
                      
    ground_config['gal'] = galaxy_config
    ground_config['input'] = catalog_config
    space_config['gal'] = galaxy_config
    space_config['input'] = catalog_config

    return [space_config, ground_config]

class InterpolationData:
# Quick container class for passing around data from these tests.  
    def __init__(config, g1_obs, g2_obs, sigma_obs, 
                 err_g1_obs=None, err_g2_obs=None, err_sigma_obs=None):
        self.g1_obs = g1_obs
        self.g2_obs = g2_obs
        self.sigma_obs = sigma_obs
        self.err_g1_obs = err_g1_obs
        self.err_g2_obs = err_g2_obs
        self.err_sigma_obs = err_sigma_obs
        if 'shear' in config['gal']:
            self.shear = config['gal']['shear']
        else:
            self.shear = 0.
        if 'magnification' in config['gal']:
            self.magnification = config['gal']['magnification']
        else:
            self.magnification = 1.
        if 'rotation' in config['gal']:
            self.angle = config['gal']['rotation']['theta']
        else:
            self.angle = 0
        self.interpolant = config['x_interpolant']
        self.padding = config['gal']['pad_factor']
        if config['psf']['type'] == 'Moffat':
            self.image_type = 'Space'
        else:
            self.image_type = 'Ground'

def test_realgalaxy(base_config, shear=None, magnification=None, angle=None, shift=None,
                    interpolant=None, padding=None, seed=None):
# Do something like Barney's compare_dft_vs_photon_config test, only do it for DFT only
# since the RealGalaxies have a Deconvolve in them.
    config = copy.deepcopy(base_config
    
    if shear:
        config['gal']['shear'] = shear
    if magnification:
        config['gal']['magnification'] = magnification
    if angle:
        config['gal']['rotation'] = {'type': 'Degrees', 'theta': 1.0*angle}
    if interpolant:    
        config['gal']['x_interpolant'] = interpolant
    if padding:
        config['gal']['pad_factor'] = padding
    if seed:
        config['image']['random_seed'] = rseed
    else:
        raise ValueError("Must pass random seed to test_realgalaxy with 'seed'")
# Things to try later:
#        config['gal']['shift'] = shift
#        config['gal']['k_interpolant'] = interpolant
    
    trial_images = galsim.config.BuildImages( nimages = config['gal']['index']['nitems'], 
        obj_num = obj_num, config = config, logger=logger, nproc=config['image']['nproc'])[0] 
    trial_results = [image.FindAdaptiveMom() for image in trial_images]
    # Get lists of g1,g2,sigma estimate (this might be quicker using a single list comprehension
    # to get a list of (g1,g2,sigma) tuples, and then unzip with zip(*), but this is clearer)
    g1obs_list.extend([res.observed_shape.g1 for res in trial_results]) 
    g2obs_list.extend([res.observed_shape.g2 for res in trial_results]) 
    sigmaobs_list.extend([res.moments_sigma for res in trial_results])

    return InterpolationData(config, g1obs=g1obs_list, g2obs=g2obs_list, sigmaobs = sigmaobs_list)

def test_realgalaxyoriginal(config, shear, dilation, rotation, shift):
# Do Barney's compare_dft_vs_photon_config test for a bunch of galaxies: want to know this behavior
# for both, if we can.
    config = copy.deepcopy(base_config
    config['gal']['type'] = 'RealGalaxyOriginal'
        
    if shear:
        config['gal']['shear'] = shear
    if magnification:
        config['gal']['magnification'] = magnification
    if angle:
        config['gal']['rotation'] = {'type': 'Degrees', 'theta': 1.0*angle}
    if interpolant:    
        config['gal']['x_interpolant'] = interpolant
    if seed:
        config['image']['random_seed'] = rseed
    else:
        raise ValueError("Must pass random seed to test_realgalaxy with 'seed'")
    
    start_index = config['gal']['index']['first']
    n_objects = config['gal']['index']['nitems']
    g1obs_list = []
    g2obs_list = []
    sigmaobs_list = []
    err_g1obs_list = []
    err_g2obs_list = []
    err_sigma_list = []
    g1obs_list_draw = []
    g2obs_list_draw = []
    sigmaobs_list_draw = []
    
    del[config['gal']['index']]
    for index_offset in range(n_objects):
        config['gal']['id'] = start_index + index_offset
        result = galsim.utilities.compare_dft_vs_photon_config(
            config, n_photons_per_trial=np, nproc=config['image']['nproc'], logger=logger,
            abs_tol_ellip=tol_ellip, abs_tol_size=tol_size, wmult = 1)
        g1obs_list_draw.append(result.g1obs_draw)
        g2obs_list_draw.append(result.g2obs_draw)
        sigmaobs_list_draw.append(result.sigma_draw)
        g1obs_list.append(result.g1obs_draw-result.delta_g1obs)
        g2obs_list.append(result.g2obs_draw-result.delta_g2obs)
        sigmaobs_list.append(result.sigma_draw-result.delta_sigma)
        err_g1obs_list.append(err_g1obs)
        err_g2obs_list.append(err_g2obs)
        err_sigma_list.append(err_sigma)
        
    return (InterpolationData(config, g1obs = g1obs_list, g2obs = g2obs_list, 
        sigmaobs = sigmaobs_list, err_g1obs = err_g1obs_list, err_g2obs = err_g2obs_list, 
        err_sigmaobs = err_sigmaobs_list), InterpolationData(config, g1obs = g1obs_list_draw, 
        g2obs = g2obs_list_draw, sigmaobs = sigmaobs_list_draw))
        
def print_results(base_answer, test_answers):
    
        

def main():
    # Define the config dictionaries we will use for all the following tests
    config_list = get_config()
    
    # Now, run through the various things we need to test in loops.
    # Right now, test rotation angles separately from shear and magnification
    # (but we can do that nested later if need be - probably with fewer tested angles).
    for base_config in config_list:                     # Ground and space
        base_answer = test_realgalaxy(base_config)
        base_answer_dft, base_answer_shoot = test_realgalaxyoriginal(base_config)
        for angle in angle_list:                        # Possible rotation angles
            for interpolant in interpolant_list:        # Possible interpolants
                for padding_type in noise_padding_list: # Noise pad or not
                    for padding in padding_list:        # Amount of padding
                        print_results(base_answer, test_realgalaxy(base_config, angle=angle, 
                                      interpolant=interpolant, seed=rseed))
                        print_results_original((base_answer_dft, base_answer_shoot), 
                                               test_realgalaxyoriginal(base_config, angle=angle, 
                                               interpolant=interpolant, seed=rseed))

        for (shear, mag) in zip(shear_list, magnification_list): # Shear and magnification
            for interpolant in interpolant_list:                 # Possible interpolants
                for padding_type in noise_padding_list:          # Noise pad or not
                    for padding in padding_list:                 # Amount of padding
                        config = copy.deepcopy(base_config) # Copy our base config and add params
                        print_results(base_answer, test_realgalaxy(base_config, shear=shear,
                                      magnification=magnification, interpolant=interpolant,
                                      seed=rseed))
                        print_results_original((base_answer_dft, base_answer_shoot), 
                                               test_realgalaxyoriginal(base_config, shear=shear,))
                                               magnification=magnification, interpolant=interpolant,
                                               seed=rseed))
                
        
if __name__ == "__main__":
    main()

