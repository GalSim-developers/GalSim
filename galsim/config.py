import collections
import os
import logging

class AttributeDict(object):
    """@brief Dictionary class that allows for easy initialization and refs to key values via
    attributes.

    NOTE: Modified a little from Jim's bot.git AttributeDict class  (Jim, please review!) so that...

    ...Tab completion now works in ipython (it didn't with the bot.git version on my build) since
    attributes are actually added to __dict__.
    
    HOWEVER this means I have redefined the __dict__ attribute to be a collections.defaultdict()
    so that Jim's previous default attrbiute behaviour is also replicated.

    I prefer this, as a newbie who uses ipython and the tab completion function to aid development,
    but does it potentially break something down the line or add undesirable behaviour? (I guessed
    not, since collections.defaultdict objects have all the usual dict() methods, but I cannot be
    sure.)
    """
    def __init__(self):
        object.__setattr__(self, "__dict__", collections.defaultdict(AttributeDict))

    def __getattr__(self, name):
        return self.__dict__[name]

    def __setattr__(self, name, value):
        self.__dict__[name] = value

    def merge(self, other):
        self.__dict__.update(other.__dict__)

    def _write(self, output, prefix=""):
        for k, v in self.__dict__.iteritems():
            if isinstance(v, AttributeDict):
                v._write(output, prefix="{0}{1}.".format(prefix, k))
            else:
                output.append("{0}{1} = {2}".format(prefix, k, repr(v)))

    def __nonzero__(self):
        return not not self.__dict__

    def __repr__(self):
        output = []
        self._write(output, "")
        return "\n".join(output)

    __str__ = __repr__


def load(config_file=None):
    """@brief Function for loading in configuration settings from the specified config file, and
    using this to augment/update values in GalSim/config/galsim_default.
    """
    config = AttributeDict()
    if (config_file != None) and (not os.path.exists(default_config_file)):
        raise IOError("User input config file "+str(config_file)+" not found")
    config.user_config_file = config_file
    thisdir, modfile = os.path.split(__file__)
    default_config_file = os.path.join(thisdir, "..", "config", "galsim_default")
    if not os.path.exists(default_config_file):
        raise IOError("Default config file galsim_default not found.")
    files = [default_config_file]
    if config_file != None:
        file.append(config_file)
    for f in reversed(files):
        execfile(f)
    return config



