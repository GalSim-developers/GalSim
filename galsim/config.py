import os
import galsim

def load(config_file=None, include_default=True):
    """Function for loading in configuration settings from the specified config file, and
    using this to augment/update values in GalSim/config/galsim_default.
    """
    config = galsim.AttributeDict()
    if (config_file != None) and (not os.path.exists(config_file)):
        raise IOError("User input config file "+str(config_file)+" not found")
    config.user_config_file = config_file
    
    if include_default:
        thisdir, modfile = os.path.split(__file__)
        default_config_file = os.path.join(thisdir, "..", "config", "galsim_default")
        if not os.path.exists(default_config_file):
            raise IOError("Default config file galsim_default not found.")
        files = [default_config_file]
    elif config_file == None:
        raise IOError("Neither default nor user input config file specified.")    
    else:
        files =[]
    if config_file != None:
        file.append(config_file)
    for f in reversed(files):
        execfile(f)
    return config



