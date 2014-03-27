Photon vs FFT

(1) GOAL:

Compare images created using the reconvolution engine with those created by directly. 
Repeat this comparison for a range of varied galsim parameters, which control the drawing engines.
This exploration will allow to find a parameter set which will guarantee accurate image rendering.
Bias on the reconvolved shape wrt to direct shape has to be smaller than m<0.0002.

(2) RUNNING THE TEST

2.1 Create a config (yaml) file. See reconvolution_validation.yaml for an example default parameter settings, which are in the main structures of the yaml file, in 'GSParams', 'gal', 'psf', 'image', etc, varied parameters are in 'vary_params'. If you don't want to repeat the measurement for direct or reconvolved images, use rebuild_direct=False or rebuild_reconv=False flags, respectively. This can be useful, for example, if varied parameters are affecting the reconvolution engine and not direct images (in this case, use rebuild_direct=False).

2.2 Run commands:
Example command 
python ~/code/GalSim/devel/external/test_reconvolution/reconvolution_validation.py reconvolution_validation.yaml 
You may want to run the defaults and varied parameters separately. 
To do it, I used flags --default_only and --vary_params_only, respectively.

2.3 Example outputs file 
results.yaml_filename.param_name.param_index.direct.cat 
results.yaml_filename.param_name.param_index.reconv.cat 
where:
param_name is the name of the varied parameter (if default set is ran, then will contain word "default" ) 
param_index - the index of a parameter in the list in the config file. 
Each row corresponds to a galaxy shape measurement in the input catalog.

(3) ANALYSING RESULTS

Use reconvolution_validation.py. This driver will go through the config file used before, find appropriate results files and then produce plots. It looks for the results files for both direct and reconvolved images: 
results.yaml_filename.param_name.param_index.direct.cat 
results.yaml_filename.param_name.param_index.revonv.cat 
However, if the pht or fft was not run (for example some parameters had rebuild_pht==False flag), the driver will look for defaults for comparison: 
results.yaml_filename.default.reconv.cat 
The script saves PNG figures.
