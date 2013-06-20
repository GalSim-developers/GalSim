"""
A test of the behavior of shear, magnification, shift, rotation, etc for different interpolants and gsparams when applied to RealGalaxy objects.
"""

import galsim
import copy
import numpy

# --- THINGS WE ARE TESTING  ---
# Interpolants
interpolant_list = ['nearest', 'sinc', 'linear', 'cubic', 'quintic', 
                    'lanczos3', 'lanczos4', 'lanczos5', 'lanczos7']
use_interpolants = interpolant_list
# Padding options
padding_list = range(2,7,2)
# Range of rotation angles
angle_list = range(0,180,15)
# Range of shifts (currently half-pixel shifts +/-x, +/-y, and the intermediate 45-degree angles)
shift_list = [galsim.PositionD(0.0,0.5),galsim.PositionD(0.5,0.0),
              galsim.PositionD(0.35,0.35),galsim.PositionD(-0.35,0.35),
              galsim.PositionD(0.0,-0.5),galsim.PositionD(-0.5,0.0),
              galsim.PositionD(-0.35,-0.35),galsim.PositionD(0.35,-0.35)]
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
                                (0.016, -0.011, 1.02), (0.020, 0.003, 0.97)]#, 
#                                ( 0.46, -0.31,  1.6), ( -0.49, -0.26,  1.4), 
#                                ( 0.47, -0.33,  1.5), ( 0.61, 0.095,  0.16)]
# --- IMPORTANT BUT NOT TESTED PARAMETERS ---
# Catalog parameters
catalog_dir = '../../examples/data'
catalog_filename = 'real_galaxy_catalog_examples.fits'
first_index = 0
pixel_scale = 0.03
nitems = 100 # How many galaxies to test
# Random seed
rseed = 999888444

# --- COMPUTATIONAL DETAILS AND FILENAMES ---
nproc = 8
original_filename = 'interpolant_test_output_original.dat'
delta_filename = 'interpolant_test_output_delta.dat'
original_file = open(original_filename,'w')
delta_file = open(delta_filename,'w')

# --- Helper functions to run the main part of the code ---
def get_config():
# A function to return three config dictionaries with the basic PSF info that we will reuse for all
# tests.
    original_config = {} # Original RealGalaxy image (before deconvolution)
    delta_config = {}
    
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
        'nproc' : nproc
    }
    delta_config['image'] = {
        'type' : 'Single',
        'pixel_scale' : pixel_scale,
        'nproc' : nproc
    }

    galaxy_config = { 'type': 'RealGalaxyOriginal', 
                      'index': { 'type': 'Sequence', 'first': first_index, 'nitems': nitems } }
    catalog_config = {'real_catalog' : { 'dir' : catalog_dir, 
        'file_name' :  catalog_filename, 'preload' : True} }
    original_config['gal'] = galaxy_config
    original_config['input'] = catalog_config
    delta_config['gal'] = galaxy_config
    delta_config['input'] = catalog_config

    field = catalog_config['real_catalog'] # Redo the 'input' processing from config
    field['type'], ignore = galsim.config.process.valid_input_types['real_catalog'][0:2]
    input_obj = galsim.config.gsobject._BuildSimple(field, 'real_catalog', catalog_config,
                                                    ignore)[0]
    original_config['real_catalog'] = input_obj
    delta_config['real_catalog'] = input_obj
    return [delta_config, original_config]

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
                    x_interpolant=None, k_interpolant=None, padding=None, seed=None, logger=None):
# Do something like Barney's compare_dft_vs_photon_config test, only do it for DFT only
# since the RealGalaxies have a Deconvolve in them.
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
        config = config, logger=logger, nproc=config['image']['nproc'])[0]
    if 'psf' in config and config['psf']['type'] is not 'Gaussian': # use EstimateShearErrors
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
                              sigmaobs=sigmaobs_list)

def print_results(g1_list, g2_list, sigma_list, test_answer, outfile=None):
    """Print the results to a file specified either via outfile kwarg or chosen using
    test_answer.image_type.
    """
    if outfile is None:
        if test_answer.image_type=='Original':
            outfile = original_file
        elif test_answer.image_type=='Delta':
            outfile = delta_file
        else:
            raise TypeError('Unknown image type in %s'%test_answer)

    if test_answer.shear[0]!=0 or test_answer.shear[1]!=0:
        # Determine expected g1 given intrinsic g1 and applied shear
        # (equations from Bernstein & Jarvis 01 (astro-ph/0107431) 
        #  which is in terms of distortions)
        intrinsic_shear_d1 = numpy.tanh(2.*numpy.arctanh(g1_list))
        intrinsic_shear_d2 = numpy.tanh(2.*numpy.arctanh(g2_list))
        applied_shears_d1 = numpy.tanh(2.*numpy.arctanh(test_answer.shear[0]))
        applied_shears_d2 = numpy.tanh(2.*numpy.arctanh(test_answer.shear[1]))
        applied_shears_mag_sq = applied_shears_d1**2+applied_shears_d2**2
        expected_d1 = (intrinsic_shear_d1+applied_shears_d1+
                       (applied_shears_d2/applied_shears_mag_sq)
                       *(1.-numpy.sqrt(1.-applied_shears_mag_sq))
                       *(intrinsic_shear_d2*applied_shears_d1-intrinsic_shear_d1*applied_shears_d2)
                      )/(1.+applied_shears_d1*intrinsic_shear_d1
                           +applied_shears_d2*intrinsic_shear_d2)
        expected_d2 = (intrinsic_shear_d2+applied_shears_d2+
                       (applied_shears_d1/applied_shears_mag_sq)
                       *(1.-numpy.sqrt(1.-applied_shears_mag_sq))
                       *(intrinsic_shear_d1*applied_shears_d2-intrinsic_shear_d2*applied_shears_d1)
                      )/(1.+applied_shears_d1*intrinsic_shear_d1
                           +applied_shears_d2*intrinsic_shear_d2)
        expected_g1 = numpy.tanh(0.5*numpy.arctanh(expected_d1))
        expected_g2 = numpy.tanh(0.5*numpy.arctanh(expected_d2))
        expected_size = numpy.sqrt(test_answer.magnification)*numpy.array(sigma_list)
    elif test_answer.angle!=0:
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
        outfile.write(str(i)+' '+str(tinterpolant_list.index(test_answer.x_interpolant))+' '+ 
            str(tinterpolant_list.index(test_answer.k_interpolant))+' '+    
            str(test_answer.padding)+' '+str(test_answer.shear[0])+' '+
            str(test_answer.shear[1])+' '+
            str(test_answer.magnification)+' '+str(test_answer.angle)+' '+
            str(test_answer.shiftx)+' '+str(test_answer.shifty)+' '+
            str(expected_g1[i])+' '+str(test_answer.g1obs[i]-expected_g1[i])+' '+
            str(expected_g2[i])+' '+str(test_answer.g2obs[i]-expected_g2[i])+' '+
            str(expected_size[i])+' '+
            str((test_answer.sigmaobs[i]-expected_size[i])/expected_size[i])+'\n')
        
def main():
    # Draw the original galaxies and measure their shapes
    rgc = galsim.RealGalaxyCatalog(catalog_filename, dir=catalog_dir)
    g1_list = []
    g2_list = []
    sigma_list = []
    for i in range(first_index, first_index+nitems):
        real_galaxy = galsim.RealGalaxy(rgc, index=i)
        real_galaxy_image = real_galaxy.original_image.draw()
        shape = CatchAdaptiveMomErrors(real_galaxy_image)
        print shape
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
    config_list = get_config()

    i=1 # For printing status statements
    
    # Now, run through the various things we need to test in loops.
    # Right now, test rotation angles separately from shear and magnification
    # (but we can do that nested later if need be - probably with fewer tested angles).
    for base_config in config_list:                     # Original galaxy; same with Gaussian PSF
        for padding in padding_list:                    # Amount of padding
            for interpolant in use_interpolants:        # Possible interpolants
                print 'Angle test ', 
                for angle in angle_list:                        # Possible rotation angles
                    print i,
                    i+=1
                    print_results(g1_list, g2_list, sigma_list, 
                                  test_realgalaxy(base_config, angle=angle, 
                                  x_interpolant=interpolant, padding=padding, seed=rseed))
                    print_results(g1_list, g2_list, sigma_list, 
                                  test_realgalaxy(base_config, angle=angle, 
                                  k_interpolant=interpolant, padding=padding, seed=rseed))
                print ''
                print 'Shear/magnification test ',
                for (g1, g2, mag) in shear_and_magnification_list: # Shear and magnification
                    print i, g1, g2,
                    i+=1
                    print_results(g1_list, g2_list, sigma_list, 
                                  test_realgalaxy(base_config, shear=(g1,g2),
                                  magnification=mag, x_interpolant=interpolant,
                                  padding=padding, seed=rseed))
                    print_results(g1_list, g2_list, sigma_list, 
                                  test_realgalaxy(base_config, shear=(g1,g2),
                                  magnification=mag, k_interpolant=interpolant,
                                  padding=padding, seed=rseed))
                print ''
                for shift in shift_list:
                    print i, shift,
                    i+=1
                    print_results(g1_list, g2_list, sigma_list, 
                                  test_realgalaxy(base_config, shift=shift, 
                                  x_interpolant=interpolant, padding=padding, seed=rseed))
                    print_results(g1_list, g2_list, sigma_list, 
                                  test_realgalaxy(base_config, shift=shift, 
                                  k_interpolant=interpolant, padding=padding, seed=rseed))
                print ''
                    

    original_file.close()
    delta_file.close()
    
if __name__ == "__main__":
    main()


