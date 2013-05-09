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
first_index = 1
nitems = 1

# Photon-shoot trial parameters, shamelessly stolen from Barney
np = int(1E5)
tol_ellip = 3.e-1
tol_size = 1.0
#tol_ellip = 3.e-5
#tol_size = 1.e-4

# Ground-image parameters
ground_fwhm = 0.65        # arcsec
ground_pixel_scale = 0.2  # arcsec
ground_lam_over_diam = 700./4.*1.E-9*206265
ground_imsize = 64

# Space-image parameters
space_lam_over_diam = 700./1.3*1.E-9*206265
space_pixel_scale = 0.05
space_imsize = 64

# Random seed
rseed = 999888444

# --- COMPUTATIONAL DETAILS AND FILENAMES ---
nproc = 4
ground_filename = 'interpolant_test_output_ground.dat'
space_filename = 'interpolant_test_output_space.dat'
ground_original_filename = 'interpolant_test_output_ground_original.dat'
space_original_filename = 'interpolant_test_output_space_original.dat'
ground_base_filename = 'interpolant_test_output_ground_base.dat'
space_base_filename = 'interpolant_test_output_space_base.dat'
ground_original_base_filename = 'interpolant_test_output_ground_original_base.dat'
space_original_base_filename = 'interpolant_test_output_space_original_base.dat'

ground_file = open(ground_filename,'w')
space_file = open(space_filename,'w')
ground_original_file = open(ground_original_filename,'w') 
space_original_file = open(space_original_filename,'w')
ground_base_file = open(ground_base_filename,'w')
space_base_file = open(space_base_filename,'w')
ground_original_base_file = open(ground_original_base_filename,'w')
space_original_base_file = open(space_original_base_filename,'w')

# --- Helper functions to run the main part of the code ---
def get_config():
# A function to return two config dictionaries with the basic PSF info that we will reuse for all
# tests.
    space_config = {}
    ground_config = {}

    space_config['psf'] = { "type" : "Airy", 'lam_over_diam' : space_lam_over_diam}
    ground_config['psf'] = { 
        'type' : 'Convolution', 
        'items' : [ {'type' : 'Kolmogorov', 'fwhm' : ground_fwhm },
                    {'type' : 'Airy', 'lam_over_diam' : ground_lam_over_diam } ]
    }
                
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
    galaxy_config = { 'type': 'RealGalaxy', 
                      'index': { 'type': 'Sequence', 'first': first_index, 'nitems': nitems } }
    catalog_config = {'real_catalog' : { 'dir' : '../../examples/data', 
        'file_name' :  'real_galaxy_catalog_example.fits', 'preload' : True} }
    ground_config['gal'] = galaxy_config
    ground_config['input'] = catalog_config
    space_config['gal'] = galaxy_config
    space_config['input'] = catalog_config

    field = catalog_config['real_catalog'] # Redo the 'input' processing from config
    field['type'], ignore = galsim.config.process.valid_input_types['real_catalog'][0:2]
    input_obj = galsim.config.gsobject._BuildSimple(field, 'real_catalog', catalog_config,
                                                    ignore)[0]
    space_config['real_catalog'] = input_obj
    ground_config['real_catalog'] = input_obj

    return [space_config, ground_config]

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
        self.interpolant = config['gal']['x_interpolant']
        self.padding = config['gal']['pad_factor']
        if config['psf']['type'] == 'Convolution':
            self.image_type = 'Ground'
        else:
            self.image_type = 'Space'

def CatchAdaptiveMomErrors(obj):
    try:
        x = obj.FindAdaptiveMom()
    except RuntimeError:
        x = -10.
    return x

def test_realgalaxy(base_config, shear=None, magnification=None, angle=None, shift=None,
                    interpolant=None, padding=None, seed=None, logger=None):
# Do something like Barney's compare_dft_vs_photon_config test, only do it for DFT only
# since the RealGalaxies have a Deconvolve in them.
    config = copy.deepcopy(base_config)
    sys.stdout.write('.')
    if shear:
        config['gal']['shear'] = {'type': 'G1G2', 'g1': shear[0], 'g2': shear[1]}
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
    pass_config = copy.deepcopy(config)
    trial_images = galsim.config.BuildImages(nimages = config['gal']['index']['nitems'], 
        config = config, logger=logger, nproc=config['image']['nproc'])[0] 
    trial_results = [CatchAdaptiveMomErrors(image) for image in trial_images]
    # Get lists of g1,g2,sigma estimate (this might be quicker using a single list comprehension
    # to get a list of (g1,g2,sigma) tuples, and then unzip with zip(*), but this is clearer)
    g1obs_list = [-10 if isinstance(res,float) else res.observed_shape.g1 for res in trial_results] 
    g2obs_list = [-10 if isinstance(res,float) else res.observed_shape.g2 for res in trial_results]
    sigmaobs_list = [-10 if isinstance(res,float) else res.moments_sigma for res in trial_results]

    return InterpolationData(config=pass_config, g1obs=g1obs_list, g2obs=g2obs_list, 
                              sigmaobs = sigmaobs_list)

def test_realgalaxyoriginal(base_config, shear=None, magnification=None, angle=None, shift=None,
                            interpolant=None, padding=None, seed=None, logger=None):
# Do Barney's compare_dft_vs_photon_config test for a bunch of galaxies: want to know this behavior
# for both, if we can.
    sys.stdout.write('.')
    config = copy.deepcopy(base_config)
    config['gal']['type'] = 'RealGalaxyOriginal'
        
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
    pass_config = copy.deepcopy(config)
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
        config['gal']['index'] = start_index + index_offset
        result = galsim.utilities.compare_dft_vs_photon_config(
            config, n_photons_per_trial=np, nproc=config['image']['nproc'], logger=logger,
            abs_tol_ellip=tol_ellip, abs_tol_size=tol_size, wmult = 1)
        g1obs_list_draw.append(result.g1obs_draw)
        g2obs_list_draw.append(result.g2obs_draw)
        sigmaobs_list_draw.append(result.sigma_draw)
        g1obs_list.append(result.g1obs_draw-result.delta_g1obs)
        g2obs_list.append(result.g2obs_draw-result.delta_g2obs)
        sigmaobs_list.append(result.sigma_draw-result.delta_sigma)
        err_g1obs_list.append(result.err_g1obs)
        err_g2obs_list.append(result.err_g2obs)
        err_sigma_list.append(result.err_sigma)
        
    return (InterpolationData(config=pass_config, g1obs = g1obs_list, g2obs = g2obs_list, 
        sigmaobs = sigmaobs_list, err_g1obs = err_g1obs_list, err_g2obs = err_g2obs_list, 
        err_sigmaobs = err_sigma_list), InterpolationData(config=pass_config, 
        g1obs = g1obs_list_draw, g2obs = g2obs_list_draw, sigmaobs = sigmaobs_list_draw))
        
def print_results(base_answer, test_answer, outfile=None):
    if test_answer.image_type == 'Ground' and base_answer.image_type == 'Ground':
        image_type = 1
    elif test_answer.image_type == 'Space' and base_answer.image_type == 'Space':
        image_type = 0
    else:
        raise RuntimeError('Trying to compare ground-based to space-based images')
    
    if ((base_answer.interpolant!=test_answer.interpolant) or 
        (base_answer.padding!=test_answer.padding)) and outfile!='base' and outfile!='Base':
            raise RuntimeError('Trying to compare different interpolants or paddings')
            
    if not outfile:
        if image_type==1:
            outfile = ground_file
        else:
            outfile = space_file
    elif outfile=='original' or outfile=='Original':
        if image_type==1:
            outfile = ground_original_file
        else:
            outfile = space_original_file
    elif outfile=='base' or outfile=='Base':
        if image_type==1:
            outfile = ground_base_file
        else:
            outfile = space_base_file
    else:
        raise ValueError('Outfile type %s not understood'%outfile)
            
    for i in range(len(test_answer.g1obs)):
        outfile.write(str(i)+' '+str(interpolant_list.index(base_answer.interpolant))+' '+    
            str(base_answer.padding)+' '+str(interpolant_list.index(test_answer.interpolant))+' '+    
            str(test_answer.padding)+' '+str(test_answer.shear[0])+' '+str(test_answer.shear[1])+
            ' '+str(test_answer.magnification)+' '+str(test_answer.angle)+' '+
            str(base_answer.g1obs[i])+' '+str(test_answer.g1obs[i]-base_answer.g1obs[i])+' '+
            str(base_answer.g2obs[i])+' '+str(test_answer.g2obs[i]-base_answer.g2obs[i])+' '+
            str(base_answer.sigmaobs[i])+' '+str(test_answer.sigmaobs[i]-base_answer.sigmaobs[i])+
            '\n')
        
def print_results_original(base_answer, test_answer, outfile=None):
    if not ((test_answer[0].image_type == test_answer[1].image_type) and
             (base_answer[0].image_type == base_answer[1].image_type)):
        raise RuntimeError('Not all input tuples have same config type')
    if test_answer[0].image_type == 'Ground' and base_answer[0].image_type == 'Ground':
        image_type = 1
    elif test_answer[0].image_type == 'Space' and base_answer[0].image_type == 'Space':
        image_type = 0
    else:
        raise RuntimeError('Trying to compare ground-based to space-based images')
    
    if ((base_answer[0].interpolant!=test_answer[0].interpolant) or 
        (base_answer[0].padding!=test_answer[0].padding)) and outfile!='base' and outfile!='Base':
            raise RuntimeError('Trying to compare different interpolants or paddings')
    if not (base_answer[0].interpolant==base_answer[1].interpolant and
             base_answer[0].padding==base_answer[1].padding and
             test_answer[0].interpolant==test_answer[1].interpolant and
             test_answer[0].padding==test_answer[1].padding):
                raise RuntimeError('Not all input tuples have same config type')
    
    if (not outfile) or outfile=='original' or outfile=='Original':
        if image_type==1:
            outfile = ground_original_file
        else:
            outfile = space_original_file
    elif outfile=='base' or outfile=='Base':
        if image_type==1:
            outfile = ground_original_base_file
        else:
            outfile = space_original_base_file
    else:
        raise ValueError('Outfile type %s not understood'%outfile)
            
    for i in range(len(test_answer[0].g1obs)):
        outfile.write(str(i)+' '+str(interpolant_list.index(base_answer[0].interpolant))+' '+    
            str(base_answer[0].padding)+' '+str(interpolant_list.index(test_answer[0].interpolant))+
            ' '+str(test_answer[0].padding)+' '+str(test_answer[0].shear[0])+' '+
            str(test_answer[0].shear[1])+' '+str(test_answer[0].magnification)+' '+
            str(test_answer[0].angle)+' '+
            str(base_answer[0].g1obs[i])+' '+
            str(base_answer[1].g1obs[i]-base_answer[0].g1obs[i])+' '+
            str(test_answer[0].g1obs[i]-base_answer[0].g1obs[i])+' '+
            str(test_answer[1].g1obs[i]-base_answer[0].g1obs[i])+' '+
            str(base_answer[0].g2obs[i])+' '+
            str(base_answer[1].g2obs[i]-base_answer[0].g2obs[i])+' '+
            str(test_answer[0].g2obs[i]-base_answer[0].g2obs[i])+' '+
            str(test_answer[1].g2obs[i]-base_answer[0].g2obs[i])+' '+
            str(base_answer[0].sigmaobs[i])+' '+
            str(base_answer[1].sigmaobs[i]-base_answer[0].sigmaobs[i])+
            str(test_answer[0].sigmaobs[i]-base_answer[0].sigmaobs[i])+
            str(test_answer[1].sigmaobs[i]-base_answer[0].sigmaobs[i])+'\n')

def main():
    # Define the config dictionaries we will use for all the following tests
    config_list = get_config()
    
    i=1 # For printing status statements
    
    # Now, run through the various things we need to test in loops.
    # Right now, test rotation angles separately from shear and magnification
    # (but we can do that nested later if need be - probably with fewer tested angles).
    for base_config in config_list:                     # Ground and space
        base_list = [] # For comparing the "base" (eg unsheared, unmagnified, unrotated) cases
        base_list_original = []
        for padding in padding_list:        # Amount of padding
            for interpolant in interpolant_list:        # Possible interpolants
                print 'Base test ', i,
                i+=1
#                base_answer = test_realgalaxy(base_config, seed=rseed, interpolant=interpolant,
#                                            padding=padding)
#                base_list.append(base_answer)
                base_answer_dft, base_answer_shoot = test_realgalaxyoriginal(base_config, 
                                            interpolant=interpolant, padding=padding, seed=rseed)
                base_list_original.append((base_answer_dft,base_answer_shoot))
                print ''
                for angle in angle_list:                        # Possible rotation angles
                    print 'Angle test ', i,
                    i+=1
#                    print_results(base_answer, test_realgalaxy(base_config, angle=angle, 
#                                  interpolant=interpolant, padding=padding, seed=rseed))
                    print_results_original((base_answer_dft, base_answer_shoot), 
                                           test_realgalaxyoriginal(base_config, angle=angle, 
                                           interpolant=interpolant, padding=padding, seed=rseed))
                    print ''
                for (g1, g2, mag) in shear_and_magnification_list: # Shear and magnification
                    print 'Shear/magnification test', i, g1, g2,
                    i+=1
#                    print_results(base_answer, test_realgalaxy(base_config, shear=(g1,g2),
#                                  magnification=mag, interpolant=interpolant,
#                                  padding=padding, seed=rseed))
                    print_results_original((base_answer_dft, base_answer_shoot), 
                                           test_realgalaxyoriginal(base_config, shear=(g1,g2),
                                           magnification=mag, interpolant=interpolant,
                                           padding=padding, seed=rseed))
                    print ''
        for k in range(len(base_list)):
            for j in range(k+1,len(base_list)):
#                print_results(base_list[k],base_list[j],outfile='base')
                print_results_original(base_list_original[k],base_list_original[j],outfile='base')

    ground_file.close()
    space_file.close()
    ground_original_file.close()
    space_original_file.close()
    ground_base_file.close()
    space_base_file.close()
    ground_original_base_file.close()
    space_original_base_file.close()
    
    
        
        
if __name__ == "__main__":
    main()

