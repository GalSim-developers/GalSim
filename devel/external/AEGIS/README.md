This README file describes the python code used to process AEGIS HST images in V and I bands to produce multi-band postage stamp images of galaxies. The selection procedure is explained in a separate document in devel/external/AEGIS/document/main.tex; a rendered version of the document may be viewed [here](https://www.overleaf.com/read/krbzsccfdbpm). 
The final products are similar to the COSMOS Real Galaxy training sample and can be found at the following [link](http://great3.jb.man.ac.uk/leaderboard/data/public/AEGIS_training_sample.tar.gz).
The script can be implemented to analyze any HST field data in multiple filters. It takes HST images in multiple bands, identifies objects with SExtractor, classifies those objects as stars and galaxies. Stars are then used to measure the PSF by comparing with precomputed Tiny Tim PSFs and identifying the focus position that most closely matches this exposure. Once the stars have been used to identify the focus position, the Tiny Tim PSF models are used for PSF estimation as a function of CCD position. Galaxies that satisfy certain criteria (see Section 3.8 in document) are selected for the main catalog. Separate catalogs are made for different bands. Each catalog can be individually used to initialize a galsim.RealGalaxyCatalog or galsim.COSMOSCatalog instance.  A postage stamp image is drawn for every galaxy in the catalog in each band. Each galaxy image will also have a postage stamp of its PSF in each band. The entries in different filter catalog files correspond to the same galaxies.

## Requirements:
### Input Files:
* Image files: Science image in fits format along with its corresponding weight map in a single folder. The input files for the multiple filters are in separate folders, the folder name being the filter name (main_path/filter/file_name).
* Co-added images : The science images in multiple bands should be added. The variance maps (1/weight map) of multiple bands should be added. Objects are detected on these Co-added images using SExtractor in dual image mode.  
* Tiny Tim Images: Images of PSF at different locations in the focal plane for different focus offsets. The PSF field files for the multiple filters should be in  separate folders.
* File with regions for manual masks. These are regions that upon visual inspection were found to contain artifacts e.g: reflection ghosts, and hence need to be masked out. The text file contains segment ID, filter, x and y coordinates of the points on a quadrilateral that marks the region to be masked, file name of science image with region to be masked. Note: A separate script identifies and masks regions near saturated stars; hence stellar masks need not be included in manual masks.
* Noise map: Fits file with the noise covariance matrix for each band. Empty regions in the science image were visually identified from which the covariance matrix of the noise was computed.(See ipython notebook check_empty_regions.ipynb)
* Other catalogs (OPTIONAL): The script also has provision to add information from other catalogs to the main catalog, e.g:redshift information from another catalog.

### Input Parameters for script:
* zero_point_mag: Zero point magnitude for each filter.
* diff_spike_params: These parameters are used to compute the size of the diffraction spikes. A polygonal mask is drawn around saturated stars to mask these spikes (see Fig1: in document). Parameters are [slope(pixels/ADU), intercept(pixels),width(pixels),angle(degrees)]. Slope and intercept relate the FLUX_AUTO of the star to the length the spike. The width of the spike is set with width. Angle gives the angle by which the polygon has to be rotated. The parameters may be different for different bands. 
* star_galaxy_params: Parameters used to separate galaxies and stars in MU_MAX Vs MAG_AUTO plot(x_div, y_div, slope). x_div gives the maximum magnitude, below which the object is saturated. y_div is the value of surface brightness per pixel for a saturated star. The slope of the line separating stars and galaxies is given by slope (See Fig 2 in document). It is recommended to run the first script get_objects.py, with mock star_galaxy_params values, on a small region and then compute the separation parameters from the objects measured. The parameters may be different for different bands.
* gain: Detector gain in e/ADU. 

### Scripts: 
The scripts are run entirely in python. Make sure the following modules are loaded:
galsim, numpy, astropy, asciidata, subprocess, os, scipy.
The detection and measurement is performed with [SExtractor](http://www.astromatic.net/software/sextractor).

## Running the script
Note: For faster computation, most scripts are written to be run on each individual image segment, so that multiple segments can be analyzed simultaneously with multiple processors.

The entire pipeline contains 6 scripts that are to be run in the order:

1. get_objects.py : Script to detect objects in a given image (segment) using SExtractor and make a catalog. Star-Galaxy separation, masking objects at segment boundaries, masking diffraction spikes and manual masks is done here. Stars for PSF estimation are also identified. 
2. remove_multi.py : Script to remove multiple detections of the same object in overlapping segments. *Note: This script is to be run once to check multiple detections over all segments.*
3. get_psf: Computes the focus offset for a given image, and uses that to estimate the PSF. Postage stamps of galaxies and PSF  are also drawn (called in get_pstamps.py).
4. clean_pstamp.py: Identifies multiple objects in the postage stamp of a galaxy and replaces the other object with noise.  
5. get_cat_seg.py: For each segment, creates a catalog with entries only for the objects with postage stamps that will appear in the main catalog. Information from other catalogs can also added in this step.
6. apply_selection.py: applies selection cuts and writes final catalog into files that can be opened with galsim.RealGalaxy (called in get_in_galsim.py)

Common functions that are called multiple times are saved in functions.py

Additional scripts are also included which were used to run the above scripts through batch jobs, for faster computation. Note: the script is written to be run on SLAC batch farm with LSF batch system. You might have to tweak it depending on how you run.

 1. additional.py : Additional code to get list of seg ids, coadd images in multiple bands and convert weight map to rms map.
 2. run_batch_first.py : Script to run get_objects.py over all segments.
 3. run_batch_second.py : Script to run get_psf.py over all segments.
 4. run_batch_third.py : Script to run run_clean_seg.py  over all segments.
 5. run_clean_seg.py : Script to run clean_pstamp.py over all postage stamps for a given segment
 6. run_batch_third_again.py : Script to run clean_pstamp.py over all postage stamps whose jobs failed.
 7. run_batch_fourth.py : Script to run get_cat_seg.py  over all segments.

## Output:
Fits files with galaxy images (in multiple bands), files with psf images (in 
multiple bands), main catalog file, selection file and fit file.
