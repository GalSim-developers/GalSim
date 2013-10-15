Photon vs FFT

(1) GOAL:

Compare photon sampled images to those rendered by FFT. 
Repeat this comparison for a range of varied galsim parameters, which control the drawing engines.
This exploration will allow to find a parameter set which will guarantee accurate image rendering.
Bias on the FFT rendered shape wrt to photon image shape has to be smaller than m<0.0002.

(2) RUNNING THE TEST

2.1 Create a config (yaml) file
See photon_vs_fft.yaml for an example
*default* parameter settings are in the main structures of the yaml file, in 
'GSParams', 'gal', 'psf', 'image', etc
*varied* parameters are in 'vary_params'
If you don't want to repeat the measurement for photon shooting for this varied parameter
(for example because the parameter controls the FFT and doesn't affect the photon shooting)
you can use rebuild_pht=False flag.

2.2 Run commands
Example command
python ~/code/GalSim/devel/external/test_photon_vs_fft/photon_vs_fft.py photon_vs_fft.ground.yaml 
You may want to run the defaults and varied parameters separately.
For example, to test the FFT parameters I had to compute the default photon images measurement 
only once. I used a cluster computer to calculate defaults for photon galaxies,
and then my desktop to calculate varied_params.
To do it, I used flags --default_only and --vary_params_only, respectively.

2.3 Example outputs file
results.yaml_filename.param_name.param_index.fft.cat 
where param_name is the name of the varied parameter 
(if default set is ran, then will contain word "default" ) 
param_index - the index of a parameter in the list in the config file.
fft or pht - results for fft or photon shooting, respectively
Each row corresponds to a galaxy shape measurement in the input catalog.

(3) ANALYSING RESULTS

Use photon_vs_fft_plots.py.
This driver will go through the config file used before, find appropriate resutls files and then 
produce plots.
It looks for the results files for both photon and FFT images:
results.yaml_filename.param_name.param_index.fft.cat 
results.yaml_filename.param_name.param_index.pht.cat 
however, if the pht or fft was not ran (for example some parameters had rebuild_pht==False flag), 
the driver will look for defaults for comparison:
results.yaml_filename.default.pht.cat 
The script saves PNG figures.








