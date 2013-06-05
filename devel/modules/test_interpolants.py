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
use_interpolants = interpolant_list
# Padding options
padding_list = range(2,7,2)
# Range of rotation angles
angle_list = range(0,180,15)
# Range of shears and magnifications to use
# This is currently the 4 largest and 4 smallest |g|s generated from a run of examples/demo11.py,
# rounded to 1-3 sig figs, plus the 4 largest blown up by a factor of 30 to really strain the 
# algorithm, and then a set of just g1/just g2/just magnification perturbations.
shear_and_magnification_list = [(0.001, 0., 0.),(0.01, 0., 0.), (0.1, 0., 0.),
                                (-0.001, 0., 0.), (-0.01, 0., 0.), (-0.1, 0., 0.),
                                (0., 0.001, 0.,),(0., 0.01, 0.), (0., 0.1, 0.),
                                (0., -0.001, 0.), (0., -0.01, 0.), (0., -0.1, 0.),
                                (0., 0., 1.01),(0., 0., 1.1), (0., 0., 1.5),
                                (0., 0., 0.99),(0., 0., 0.9), (0., 0., 0.5),
                                (0.0003, 0.0003, 1.01), (0.0005, 0.0001, 0.96), 
                                (-0.0004, 0.0009, 1.03), (0.001, -0.0003, 0.996), 
                                (0.015, -0.010, 1.02), (-0.016, -0.009, 1.01), 
                                (0.016, -0.011, 1.02), (0.020, 0.003, 0.97), 
                                ( 0.46, -0.31,  1.6), ( -0.49, -0.26,  1.4), 
                                ( 0.47, -0.33,  1.5), ( 0.61, 0.095,  0.16)]


# --- IMPORTANT BUT NOT TESTED PARAMETERS ---

# Galaxy indices
first_index = 0
nitems = 100 # Currently, do all the files in the sample catalog

# Ground-image parameters
ground_fwhm = 0.65        # arcsec
ground_pixel_scale = 0.2  # arcsec
# 700 nm wavelength, 4 m telescope, convert to radians then arcsec
ground_lam_over_diam = 700./4.*1.E-9*206265 
ground_imsize = 32

# Space-image parameters
space_lam_over_diam = 700./1.3*1.E-9*206265
space_pixel_scale = 0.05
space_imsize = 32

# Random seed
rseed = 999888444

# --- COMPUTATIONAL DETAILS AND FILENAMES ---
nproc = 8
ground_filename = 'interpolant_test_output_ground.dat'
space_filename = 'interpolant_test_output_space.dat'
original_filename = 'interpolant_test_output_original.dat'
ground_base_filename = 'interpolant_test_output_ground_base.dat'
space_base_filename = 'interpolant_test_output_space_base.dat'
original_base_filename = 'interpolant_test_output_original_base.dat'

ground_file = open(ground_filename,'w')
space_file = open(space_filename,'w')
original_file = open(original_filename,'w')
ground_base_file = open(ground_base_filename,'w')
space_base_file = open(space_base_filename,'w')
original_base_file = open(original_base_filename,'w')

# --- Helper functions to run the main part of the code ---
def get_config():
# A function to return four config dictionaries with the basic PSF info that we will reuse for all
# tests.
    space_config = {}    # Space-like data
    ground_config = {}   # Ground-like data
    original_config = {} # Original RealGalaxy image (before deconvolution)
    
    space_config['psf'] = { "type" : "Airy", 'lam_over_diam' : space_lam_over_diam}
    ground_config['psf'] = { 
        'type' : 'Convolution', 
        'items' : [ {'type' : 'Kolmogorov', 'fwhm' : ground_fwhm },
                    {'type' : 'Airy', 'lam_over_diam' : ground_lam_over_diam } ]
    }
    # no psf for nopsf_config and original_config
    original_config['pix'] = {'type': 'None'}
                
    space_config['image'] = {
        'type' : 'Single',
        'size' : space_imsize,
        'pixel_scale' : space_pixel_scale,
        'nproc' : nproc
    }
    ground_config['image'] = {
        'type' : 'Single',
        'size' : ground_imsize,
        'pixel_scale' : ground_pixel_scale,
        'nproc' : nproc
    }
    original_config['image'] = {
        'type' : 'Single',
        'size' : space_imsize,
        'pixel_scale' : space_pixel_scale,
        'nproc' : nproc
    }

    galaxy_config = { 'type': 'RealGalaxy', 
                      'index': { 'type': 'Sequence', 'first': first_index, 'nitems': nitems } }
    catalog_config = {'real_catalog' : { 'dir' : '../../examples/data', 
        'file_name' :  'real_galaxy_catalog_example.fits', 'preload' : True} }
    ground_config['gal'] = galaxy_config
    ground_config['input'] = catalog_config
    space_config['gal'] = galaxy_config
    space_config['input'] = catalog_config
    galaxy_config['type'] = 'RealGalaxyOriginal'
    original_config['gal'] = galaxy_config
    original_config['input'] = catalog_config

    field = catalog_config['real_catalog'] # Redo the 'input' processing from config
    field['type'], ignore = galsim.config.process.valid_input_types['real_catalog'][0:2]
    input_obj = galsim.config.gsobject._BuildSimple(field, 'real_catalog', catalog_config,
                                                    ignore)[0]
    space_config['real_catalog'] = input_obj
    ground_config['real_catalog'] = input_obj
    original_config['real_catalog'] = input_obj
    return [original_config]#, space_config, ground_config]

class InterpolationData:
# Quick container class for passing around data from these tests.  
    def __init__(self, config, g1obs=None, g2obs=None, sigmaobs=None, 
                 err_g1obs=None, err_g2obs=None, err_sigmaobs=None):
        self.g1obs = g1obs
        self.g2obs = g2obs
        self.sigmaobs = sigmaobs
        self.err_g1obs = err_g1obs
        self.err_g2obs = err_g2obs
        self.err_sigmaobs = err_sigmaobs
        if 'shear' in config['gal']:
            self.shear = [config['gal']['shear']['g1'],config['gal']['shear']['g2']]
        else:
            self.shear = [0., 0.]
        if 'magnification' in config['gal']:
            self.magnification = config['gal']['magnification']
        else:
            self.magnification = 1.
        if 'rotation' in config['gal']:
            self.angle = config['gal']['rotation']['theta']
        else:
            self.angle = 0
        if 'x_interpolant' in config['gal']:
            self.x_interpolant = config['gal']['x_interpolant']
        else:  
            self.x_interpolant = 'default'
        if 'k_interpolant' in config['gal']:
            self.k_interpolant = config['gal']['k_interpolant']
        else:  
            self.k_interpolant = 'default'
        self.padding = config['gal']['pad_factor']
        if 'psf' not in config:
            self.image_type = 'Original'
        elif config['psf']['type'] == 'Convolution':
            self.image_type = 'Ground'
        else:
            self.image_type = 'Space'

# Don't fail the entire program if a shape cannot be determined...
def CatchAdaptiveMomErrors(obj):
    try:
        x = obj.FindAdaptiveMom()
    except RuntimeError:
        x = -10.
    return x
    
def CatchEstimateShearErrors(obj, psf):
    try:
        x = galsim.hsm.EstimateShear(obj,psf)
    except:
        x = -10.
    return x

def test_realgalaxy(base_config, shear=None, magnification=None, angle=None, shift=None,
                    x_interpolant=None, k_interpolant=None, padding=None, seed=None, logger=None):
# Do something like Barney's compare_dft_vs_photon_config test, only do it for DFT only
# since the RealGalaxies have a Deconvolve in them.
    config = copy.deepcopy(base_config)
    sys.stdout.write('.')
    # Add to the config dictionary any requested parameters
    if shear:
        config['gal']['shear'] = {'type': 'G1G2', 'g1': shear[0], 'g2': shear[1]}
    if magnification:
        config['gal']['magnification'] = magnification
    if angle:
        config['gal']['rotation'] = {'type': 'Degrees', 'theta': 1.0*angle}
    if x_interpolant:    
        config['gal']['x_interpolant'] = x_interpolant
    if k_interpolant:    
        config['gal']['k_interpolant'] = k_interpolant
    if padding:
        config['gal']['pad_factor'] = padding
    if seed:
        config['image']['random_seed'] = rseed
    else:
        raise ValueError("Must pass random seed to test_realgalaxy with 'seed'")
    pass_config = copy.deepcopy(config) # To pass to the InterpolationData routine
    trial_images = galsim.config.BuildImages(nimages = config['gal']['index']['nitems'], 
        config = config, logger=logger, nproc=config['image']['nproc'])[0]
    if 'psf' in config: # use EstimateShearErrors
        config = copy.deepcopy(base_config)
        del config['gal']
        psf_image = galsim.config.BuildImage(config = config)[0]
        trial_results = [CatchEstimateShearErrors(trial_image,psf_image) 
                                                             for trial_image in trial_images] 
    else:               # use AdaptiveMoments
        trial_results = [CatchAdaptiveMomErrors(trial_image) for trial_image in trial_images]
    g1obs_list = [-10 if isinstance(res,float) else res.observed_shape.g1 
                                                                    for res in trial_results] 
    g2obs_list = [-10 if isinstance(res,float) else res.observed_shape.g2 for res in trial_results]
    sigmaobs_list = [-10 if isinstance(res,float) else res.moments_sigma for res in trial_results]

    return InterpolationData(config=pass_config, g1obs=g1obs_list, g2obs=g2obs_list, 
                              sigmaobs = sigmaobs_list)

def print_results(base_answer, test_answer, outfile=None):
    if outfile and outfile.lower()!='base':
        raise ValueError('Outfile type %s not understood'%outfile)

    if ((base_answer.x_interpolant!=test_answer.x_interpolant) or 
        (base_answer.k_interpolant!=test_answer.k_interpolant) or
        (base_answer.padding!=test_answer.padding)) and outfile!='base' and outfile!='Base':
            raise RuntimeError('Trying to compare different interpolants or paddings')
            
    # Test that both images use the same PSF
    if test_answer.image_type == 'Ground' and base_answer.image_type == 'Ground':
        if not outfile:
            outfile = ground_file
        else:
            outfile = ground_base_file
    elif test_answer.image_type == 'Space' and base_answer.image_type == 'Space':
        if not outfile:
            outfile = space_file
        else:
            outfile = space_base_file
    elif test_answer.image_type == 'Original' and base_answer.image_type == 'Original':
        if not outfile:
            outfile = original_file
        else:
            outfile = original_base_file
    else:
        raise RuntimeError('Trying to compare images with different PSFs')
    
    # Since we didn't want to cycle through 'default', but it's a used option...
    tinterpolant_list = interpolant_list+['default'] 
    # Write everything out as a number, so it can be loaded into python with numpy.loadtxt
    # (which yells at you if you use strings)
    for i in range(len(test_answer.g1obs)):
        outfile.write(str(i)+' '+str(tinterpolant_list.index(base_answer.x_interpolant))+' '+ 
            str(tinterpolant_list.index(base_answer.k_interpolant))+' '+    
            str(base_answer.padding)+' '+str(tinterpolant_list.index(test_answer.x_interpolant))
            +' '+str(tinterpolant_list.index(test_answer.k_interpolant))+' '+     
            str(test_answer.padding)+' '+str(test_answer.shear[0])+' '+str(test_answer.shear[1])+
            ' '+str(test_answer.magnification)+' '+str(test_answer.angle)+' '+
            str(base_answer.g1obs[i])+' '+str(test_answer.g1obs[i]-base_answer.g1obs[i])+' '+
            str(base_answer.g2obs[i])+' '+str(test_answer.g2obs[i]-base_answer.g2obs[i])+' '+
            str(base_answer.sigmaobs[i])+' '+str(test_answer.sigmaobs[i]-base_answer.sigmaobs[i])+
            '\n')
        

def main():
    # Define the config dictionaries we will use for all the following tests
    config_list = get_config()
    
    i=1 # For printing status statements
    
    # Now, run through the various things we need to test in loops.
    # Right now, test rotation angles separately from shear and magnification
    # (but we can do that nested later if need be - probably with fewer tested angles).
    for base_config in config_list:                     # Ground, space, no PSF, original galaxy
        base_list = [] # For comparing the "base" (ie unsheared, unmagnified, unrotated) cases
        for padding in padding_list:                    # Amount of padding
            for interpolant in use_interpolants:        # Possible interpolants
                print 'Base test ', i,
                i+=1
                base_answer_x = test_realgalaxy(base_config, seed=rseed, x_interpolant=interpolant,
                                            padding=padding)
                base_list.append(base_answer_x)
                base_answer_k = test_realgalaxy(base_config, seed=rseed, k_interpolant=interpolant,
                                            padding=padding)
                base_list.append(base_answer_k)
                print '' # Start a new line for each status printout
                for angle in angle_list:                        # Possible rotation angles
                    print 'Angle test ', i,
                    i+=1
                    print_results(base_answer_x, test_realgalaxy(base_config, angle=angle, 
                                  x_interpolant=interpolant, padding=padding, seed=rseed))
                    print_results(base_answer_k, test_realgalaxy(base_config, angle=angle, 
                                  k_interpolant=interpolant, padding=padding, seed=rseed))
                    print ''
                for (g1, g2, mag) in shear_and_magnification_list: # Shear and magnification
                    print 'Shear/magnification test', i, g1, g2,
                    i+=1
                    print_results(base_answer_x, test_realgalaxy(base_config, shear=(g1,g2),
                                  magnification=mag, x_interpolant=interpolant,
                                  padding=padding, seed=rseed))
                    print_results(base_answer_k, test_realgalaxy(base_config, shear=(g1,g2),
                                  magnification=mag, k_interpolant=interpolant,
                                  padding=padding, seed=rseed))
                    print ''
        for k in range(len(base_list)):
            print_results(base_list[k],base_list[k],outfile='base')

    ground_file.close()
    space_file.close()
    original_file.close()
    ground_base_file.close()
    space_base_file.close()
    original_file.close()
    
if __name__ == "__main__":
    main()

