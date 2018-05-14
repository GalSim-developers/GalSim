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

"""
A test of the behavior of shear, magnification, shift, rotation, etc for different interpolants and
gsparams when applied to RealGalaxy objects.

This file (or test_interpolants_parametric.py, which calls the print function of this script) is 
meant to be used with plot_test_interpolants.py.  The column mapping is hard-coded into both the
print function here and the main function of plot_test_interpolants.py, so changes to the column
structure here should be replicated there.
"""

import galsim
import copy
import numpy

# --- THINGS WE ARE TESTING  ---
# Interpolants
interpolant_list = ['cubic', 'quintic', 
                    'lanczos3', 'lanczos4', 'lanczos5', 'lanczos7']
# use_interpolants may be overwritten here or in main() to only test some of the above interpolants
use_interpolants = interpolant_list
# Padding options.  We do not try pad_factor=2 as preliminary testing showed that it was
# significantly more biased than 4 or 6: see e.g. the plots in the comment at
# https://github.com/GalSim-developers/GalSim/issues/389#issuecomment-20386454 .
padding_list = range(4,7,2)
# Range of rotation angles
angle_list = range(0,180,15)
# Range of shifts (currently half-pixel shifts +/-x, +/-y, and the intermediate 45-degree angles)
shift_list = [galsim.PositionD(0.0,0.5),galsim.PositionD(0.5,0.0),
              galsim.PositionD(0.35,0.35),galsim.PositionD(-0.35,0.35),
              galsim.PositionD(0.0,-0.5),galsim.PositionD(-0.5,0.0),
              galsim.PositionD(-0.35,-0.35),galsim.PositionD(0.35,-0.35)]
# Range of shears and magnifications to use
# This is currently the 4 largest and 4 smallest |g|s generated from a run of examples/demo11.py,
# rounded to 1-3 sig figs, and then a set of just g1/just g2/just magnification perturbations.
shear_and_magnification_list =  [(0.05, 0., 1.),(0.03, 0., 1.), (0.01, 0., 1.),
                                 (-0.01, 0., 1.), (-0.03, 0., 1.), (-0.05, 0., 1.),
                                 (0., 0.05, 1.,),(0., 0.03, 1.), (0., 0.01, 1.),
                                 (0., -0.01, 1.), (0., -0.03, 1.), (0., -0.05, 1.),
                                 (0., 0., 1.01),(0., 0., 1.03), (0., 0., 1.1),
                                 (0., 0., 0.99),(0., 0., 0.97), (0., 0., 0.9),
                                 (0.0003, 0.0003, 1.01), (0.0005, 0.0001, 0.96), 
                                 (-0.0004, 0.0009, 1.03), (0.001, -0.0003, 0.996), 
                                 (0.015, -0.010, 1.02), (-0.016, -0.009, 1.01), 
                                 (0.016, -0.011, 1.02), (0.020, 0.003, 0.97)]
# --- IMPORTANT BUT NOT TESTED PARAMETERS ---
# Catalog parameters
catalog_dir = '../../examples/data'
catalog_filename = 'real_galaxy_catalog_23.5_example.fits'
default_first_index = 0 # Note: both of these may be superceded via command-line arguments!
default_nitems = 100 # How many galaxies to test
pixel_scale = 0.03 # COSMOS pixel scale
imsize = 512
# Random seed
rseed = 999888444

# --- COMPUTATIONAL DETAILS AND FILENAMES ---
nproc = 8
default_file_root = 'interpolant_test_output_' # May be overwritten on the command line

# --- Helper functions to run the main part of the code ---
def get_config(nitems=default_nitems,first_index=default_first_index,file_root=default_file_root):
    """
    A function to return the config dictionaries with the basic PSF info and parameters that we will 
    reuse for all tests.
    """
    original_config = {}
    delta_config = {}
    original_filename = file_root+'original.dat'
    delta_filename = file_root+'delta.dat'
    
    delta_config['psf'] = {
        'type' : 'Gaussian',
        'sigma': 1.E-8
    }
    # no pixel convolution for original_config or delta_config (it's already in there)
    original_config['pix'] = {'type': 'None'}
    delta_config['pix'] = {'type': 'None'}
                
    original_config['image'] = {
        'type' : 'Single',
        'pixel_scale' : pixel_scale,
        'size': imsize,
        'nproc' : nproc
    }
    delta_config['image'] = {
        'type' : 'Single',
        'pixel_scale' : pixel_scale,
        'size': imsize,
        'nproc' : nproc
    }

    galaxy_config = { 'type': 'RealGalaxyOriginal', 
                      'index': { 'type': 'Sequence', 'first': first_index, 'nitems': nitems },
                      'noise_pad_size': imsize*pixel_scale, 'whiten': True}
    catalog_config = {'real_catalog' : { 'dir' : catalog_dir, 
                                         'file_name' :  catalog_filename, 'preload' : False} }
    original_config['gal'] = galaxy_config
    original_config['input'] = catalog_config
    delta_config['gal'] = galaxy_config
    delta_config['input'] = catalog_config

    field = catalog_config['real_catalog'] # Redo the 'input' processing from config
    field['type'], ignore = galsim.config.process.valid_input_types['real_catalog'][0:2]
    input_obj = [ galsim.config.gsobject._BuildSimple(field, 'real_catalog', catalog_config,
                                                    ignore)[0] ]
    original_config['real_catalog'] = input_obj
    delta_config['real_catalog'] = input_obj
    # original_config doesn't do the convolution by a tiny-Gaussian pseudo-delta function, which
    # means the k-space interpolants may have no effect.  It is here for legacy reasons, but it's
    # probably best to just use the delta versions.
    return [(delta_config,delta_filename)]#, (original_config,original_filename)]

class InterpolationData:
    """ Quick container class for passing around data from the interpolation tests.
    """
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
        if 'shift' in config['gal']:
            self.shiftx = config['gal']['shift'].x
            self.shifty = config['gal']['shift'].y
        else:
            self.shiftx = 0
            self.shifty = 0
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
        elif config['psf']['type'] == 'Gaussian':
            self.image_type = 'Delta'
        else:
            raise TypeError('Cannot detect image type from config dict in InterpolationData '
                              'initialization.')

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
                    x_interpolant=None, k_interpolant=None, padding=None, seed=None):
    # Do something like Barney's compare_dft_vs_photon_config test
    config = copy.deepcopy(base_config)
    # Add any requested parameters to the config dictionary
    if shear:
        config['gal']['shear'] = {'type': 'G1G2', 'g1': shear[0], 'g2': shear[1]}
    if magnification:
        config['gal']['magnification'] = magnification
    if angle:
        config['gal']['rotation'] = {'type': 'Degrees', 'theta': 1.0*angle}
    if shift:
        config['gal']['shift'] = config['image']['pixel_scale']*shift
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
        config = config, nproc=config['image']['nproc'])[0]
    if 'psf' in config and config['psf']['type'] is not 'Gaussian': # use EstimateShearErrors
        config = copy.deepcopy(base_config)
        del config['gal']
        psf_image = galsim.config.BuildImage(config = config)[0]
        trial_results = [CatchEstimateShearErrors(trial_image,psf_image) 
                                                             for trial_image in trial_images] 
    else: # use AdaptiveMoments
        trial_results = [CatchAdaptiveMomErrors(trial_image) for trial_image in trial_images]
    g1obs_list = [-10 if isinstance(res,float) else res.observed_shape.g1 for res in trial_results] 
    g2obs_list = [-10 if isinstance(res,float) else res.observed_shape.g2 for res in trial_results]
    sigmaobs_list = [-10 if isinstance(res,float) else res.moments_sigma for res in trial_results]
    return InterpolationData(config=pass_config, g1obs=g1obs_list, g2obs=g2obs_list, 
                              sigmaobs=sigmaobs_list)

def print_results(f, g1_list, g2_list, sigma_list, test_answer, first_index=0):
    """Print the results to a file specified either via outfile kwarg or chosen using
    test_answer.image_type.
    
    If you intend to use this with plot_test_interpolants.py, and you change the column order,
    make sure to change it in plot_test_interpolants.py as well--it's hard-coded both places!
    """
    if test_answer.shear[0]!=0 or test_answer.shear[1]!=0: 
        # If there's an applied shear, compute the value of the expected shear
        test_shear = galsim.Shear(g1=test_answer.shear[0], g2=test_answer.shear[1]) 
        expected_shears = [test_shear+galsim.Shear(g1=tg[0], g2=tg[1])
                                if (tg[0]!=-10 and tg[1]!=-10) else -10 
                                for tg in zip(g1_list,g2_list)]
        expected_g1 = [e.g1 if isinstance(e,galsim.Shear) else -10 for e in expected_shears]
        expected_g2 = [e.g2 if isinstance(e,galsim.Shear) else -10 for e in expected_shears]
        expected_size = numpy.sqrt(test_answer.magnification)*numpy.array(sigma_list)
    elif test_answer.angle!=0:
        # If there's an applied rotation, rotate the truth shears as well
        sin2theta = numpy.sin(2.*numpy.pi/180.*test_answer.angle) 
        cos2theta = numpy.cos(2.*numpy.pi/180.*test_answer.angle)
        expected_g1 = g1_list*cos2theta-g2_list*sin2theta
        expected_g2 = g1_list*sin2theta+g2_list*cos2theta
        expected_size = sigma_list
    else:
        expected_g1 = g1_list
        expected_g2 = g2_list
        expected_size = sigma_list
    # Since we didn't want to cycle through 'default', but it's a used option, add it to the list
    tinterpolant_list = interpolant_list+['default'] 
    # Write everything out as a number, so it can be loaded into python with numpy.loadtxt
    # (which yells at you if you use strings)
    for i in range(len(test_answer.g1obs)):
        f.write(str(i+first_index)+' '+str(tinterpolant_list.index(test_answer.x_interpolant))+' '+ 
            str(tinterpolant_list.index(test_answer.k_interpolant))+' '+    
            str(test_answer.padding)+' '+str(test_answer.shear[0])+' '+
            str(test_answer.shear[1])+' '+
            str(test_answer.magnification)+' '+str(test_answer.angle)+' '+
            str(test_answer.shiftx)+' '+str(test_answer.shifty)+' '+
            str(g1_list[i])+' '+str(expected_g1[i])+' '+str(test_answer.g1obs[i]-expected_g1[i])+' '+
            str(g2_list[i])+' '+str(expected_g2[i])+' '+str(test_answer.g2obs[i]-expected_g2[i])+' '+
            str(sigma_list[i])+' '+str(expected_size[i])+' '+
            str((test_answer.sigmaobs[i]-expected_size[i])/expected_size[i])+'\n')
        
def main(args):
    # Draw the original galaxies and measure their shapes
    rgc = galsim.RealGalaxyCatalog(catalog_filename, dir=catalog_dir)

    g1_list = []
    g2_list = []
    sigma_list = []
    for i in range(args.first_index, args.first_index+args.nitems):
        test_image = galsim.ImageD(imsize,imsize)
        real_galaxy = galsim.RealGalaxy(rgc, index=i)
        real_galaxy.original_image.draw(test_image)
        shape = CatchAdaptiveMomErrors(test_image)
        if shape==-10:
            g1_list.append(-10)
            g2_list.append(-10)
            sigma_list.append(-10)
        else:
            g1_list.append(shape.observed_shape.g1)
            g2_list.append(shape.observed_shape.g2)
            sigma_list.append(shape.moments_sigma)
    g1_list = numpy.array(g1_list)
    g2_list = numpy.array(g2_list)
    
    # Define the config dictionaries we will use for all the following tests
    config_and_file_list = get_config(nitems=args.nitems,first_index=args.first_index,
                                      file_root=args.file_root)

    i=1 # For printing status statements
    
    # Now, run through the various things we need to test in loops.
    for base_config, output_file in config_and_file_list: 
        f = open(output_file,'w')
        for padding in padding_list:                            # Amount of padding
            for interpolant in use_interpolants:                # Possible interpolants
                print 'Angle test ', 
                for angle in angle_list:                        # Possible rotation angles
                    print i,
                    i+=1
                    print_results(f, g1_list, g2_list, sigma_list, 
                                  test_realgalaxy(base_config, angle=angle,
                                    x_interpolant=interpolant, padding=padding, 
                                    seed=rseed+args.first_index),
                                  first_index=args.first_index)
                    print_results(f, g1_list, g2_list, sigma_list, 
                                  test_realgalaxy(base_config, angle=angle,
                                    k_interpolant=interpolant, padding=padding, 
                                    seed=rseed+args.first_index),
                                  first_index=args.first_index)
                print ''
                print 'Shear/magnification test ',
                for (g1, g2, mag) in shear_and_magnification_list: # Shear and magnification
                    print i, "(", g1, g2, ")",
                    i+=1
                    print_results(f, g1_list, g2_list, sigma_list, 
                                  test_realgalaxy(base_config, shear=(g1,g2),
                                    magnification=mag, x_interpolant=interpolant,
                                    padding=padding, seed=rseed+args.first_index),
                                  first_index=args.first_index)
                    print_results(f, g1_list, g2_list, sigma_list, 
                                  test_realgalaxy(base_config, shear=(g1,g2),
                                    magnification=mag, k_interpolant=interpolant,
                                    padding=padding, seed=rseed+args.first_index),
                                  first_index=args.first_index)
                print ''
                for shift in shift_list:
                    print i, "(", shift, ")",
                    i+=1
                    print_results(f, g1_list, g2_list, sigma_list, 
                                  test_realgalaxy(base_config, shift=shift,
                                    x_interpolant=interpolant, padding=padding, 
                                    seed=rseed+args.first_index),
                                  first_index=args.first_index)
                    print_results(f, g1_list, g2_list, sigma_list, 
                                  test_realgalaxy(base_config, shift=shift,
                                    k_interpolant=interpolant, padding=padding, 
                                    seed=rseed+args.first_index),
                                  first_index=args.first_index)
                print ''
        f.close()            
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run a suite of GalSim interpolant tests.')
    parser.add_argument('-n','--number-of-objects', 
                        help='Number of objects to run tests on (default: %i)'%default_nitems,
                        default=default_nitems, type=int, dest='nitems')
    parser.add_argument('-f','--first-item', 
                        help='Index of first galaxy in sequence to run tests on '
                             '(default: %i)'%default_first_index,
                        default=default_first_index, type=int, dest='first_index')
    parser.add_argument('-o','--output-file-root', 
                        help='Root name of output files (default: %s)'%default_file_root,
                        default=default_file_root, type=str, dest='file_root')
    args = parser.parse_args()
    main(args)
