# Copyright 2012, 2013 The GalSim developers:
# https://github.com/GalSim-developers
#
# This file is part of GalSim: The modular galaxy image simulation toolkit.
#
# GalSim is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# GalSim is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with GalSim.  If not, see <http://www.gnu.org/licenses/>
#
"""@file catalog.py
Routines for controlling catalog Input/Output with GalSim. 
"""

import galsim

class Catalog(object):
    """A class storing the data from an input catalog.

    Each row corresponds to a different object to be built, and each column stores some item of
    information about that object (e.g. flux or half_light_radius).

    After construction, the following fields are available:

        self.nobjects   The number of objects in the catalog.
        self.ncols      The number of columns in the catalog.
        self.isfits     Whether the catalog is a fits catalog.
        self.names      For a fits catalog, the valid column names.


    @param file_name     Filename of the input catalog. (Required)
    @param dir           Optionally a directory name can be provided if the file_name does not 
                         already include it.
    @param file_type     Either 'ASCII' or 'FITS'.  If None, infer from the file name ending.
                         (default `file_type = None`)
    @param comments      The character used to indicate the start of a comment in an
                         ASCII catalog.  (default `comments='#'`)
    @param hdu           Which hdu to use for FITS files.  (default `hdu = 1`)
    """
    _req_params = { 'file_name' : str }
    _opt_params = { 'dir' : str , 'file_type' : str , 'comments' : str , 'hdu' : int }
    _single_params = []
    _takes_rng = False

    # nobjects_only is an intentionally undocumented kwarg that should be used only by
    # the config structure.  It indicates that all we care about is the nobjects parameter.
    # So skip any other calculations that might normally be necessary on construction.
    def __init__(self, file_name, dir=None, file_type=None, comments='#', hdu=1,
                 nobjects_only=False):

        # First build full file_name
        self.file_name = file_name.strip()
        if dir:
            import os
            self.file_name = os.path.join(dir,self.file_name)
    
        if not file_type:
            if self.file_name.lower().endswith('.fits'):
                file_type = 'FITS'
            else:
                file_type = 'ASCII'
        file_type = file_type.upper()
        if file_type not in ['FITS', 'ASCII']:
            raise ValueError("file_type must be either FITS or ASCII if specified.")
        self.file_type = file_type

        if file_type == 'FITS':
            self.read_fits(hdu, nobjects_only)
        else:
            self.read_ascii(comments, nobjects_only)
            
    def read_ascii(self, comments, nobjects_only):
        """Read in an input catalog from an ASCII file.
        """
        # If all we care about is nobjects, this is quicker:
        if nobjects_only:
            # See the script devel/testlinecounting.py that tests several possibilities.
            # An even faster version using buffering is possible although it requires some care
            # around edge cases, so we use this one instead, which is "correct by inspection".
            f = open(self.file_name)
            if (len(comments) == 1):
                c = comments[0]
                self.nobjects = sum(1 for line in f if line[0] != c)
            else:
                self.nobjects = sum(1 for line in f if not line.startswith(comments))
            return

        import numpy
        # Read in the data using the numpy convenience function
        # Note: we leave the data as str, rather than convert to float, so that if
        # we have any str fields, they don't give an error here.  They'll only give an 
        # error if one tries to convert them to float at some point.
        self.data = numpy.loadtxt(self.file_name, comments=comments, dtype=str)
        # If only one row, then the shape comes in as one-d.
        if len(self.data.shape) == 1:
            self.data = self.data.reshape(1, -1)
        if len(self.data.shape) != 2:
            raise IOError('Unable to parse the input catalog as a 2-d array')

        self.nobjects = self.data.shape[0]
        self.ncols = self.data.shape[1]
        self.isfits = False

    def read_fits(self, hdu, nobjects_only):
        """Read in an input catalog from a FITS file.
        """
        import pyfits
        raw_data = pyfits.getdata(self.file_name, hdu)
        if pyfits.__version__ > '3.0':
            self.names = raw_data.columns.names
        else:
            self.names = raw_data.dtype.names
        self.nobjects = len(raw_data.field(self.names[0]))
        if (nobjects_only): return
        # The pyfits raw_data is a FITS_rec object, which isn't picklable, so we need to 
        # copy the fields into a new structure to make sure our Catalog is picklable.
        # The simplest is probably a dict keyed by the field names, which we save as self.data.
        self.data = {}
        for name in self.names:
            self.data[name] = raw_data.field(name)
        self.ncols = len(self.names)
        self.isfits = True

    def get(self, index, col):
        """Return the data for the given index and col in its native type.

        For ASCII catalogs, col is the column number.  
        For FITS catalogs, col is a string giving the name of the column in the FITS table.

        Also, for ASCII catalogs, the "native type" is always str.  For FITS catalogs, it is 
        whatever type is specified for each field in the binary table.
        """
        if self.isfits:
            if col not in self.names:
                raise KeyError("Column %s is invalid for catalog %s"%(col,self.file_name))
            if index < 0 or index >= self.nobjects:
                raise IndexError("Object %d is invalid for catalog %s"%(index,self.file_name))
            if index >= len(self.data[col]):
                raise IndexError("Object %d is invalid for column %s"%(index,col))
            return self.data[col][index]
        else:
            icol = int(col)
            if icol < 0 or icol >= self.ncols:
                raise IndexError("Column %d is invalid for catalog %s"%(icol,self.file_name))
            if index < 0 or index >= self.nobjects:
                raise IndexError("Object %d is invalid for catalog %s"%(index,self.file_name))
            return self.data[index, icol]

    def getFloat(self, index, col):
        """Return the data for the given index and col as a float if possible
        """
        return float(self.get(index,col))

    def getInt(self, index, col):
        """Return the data for the given index and col as an int if possible
        """
        return int(self.get(index,col))


class Dict(object):
    """A class that reads a python dict from a file

    After construction, it behaves like a regular python dict, with one exception.
    In order to facilitate getting values in a hierarchy of fields, we allow the '.'
    character to chain keys together for the get method.  So,

        d.get('noise.properties.variance')

    is expanded into

        d['noise']['properties']['variance'] 

    Furthermore, if a "key" is really an integer, then it is used as such, which accesses 
    the corresponding element in a list.  e.g.

        d.get('noise_models.2.variance')
        
    is equivalent to 

        d['noise_models'][2]['variance']

    This makes it much easier to access arbitrary elements within parameter files.

    Caveat: The above prescription means that an element whose key really has a '.' in it
    won't be accessed correctly.  This is probably a rare occurrence, but the workaround is
    to set `key_split` to a different character or string and use that to chain the keys.


    @param file_name     Filename storing the dict. (Required)
    @param dir           Optionally a directory name can be provided if the file_name does not 
                         already include it.
    @param file_type     Options are 'Pickle', 'YAML', or 'JSON' or None.  If None, infer from 
                         the file name extension ('.p*', '.y*', '.j*' respectively).
                         (default `file_type = None`)
    @param key_split     The character (or string) to use to split chained keys.  (c.f. the 
                         description of this feature above.)  (default `key_split = '.'`)
    """
    _req_params = { 'file_name' : str }
    _opt_params = { 'dir' : str , 'file_type' : str, 'key_split' : str }
    _single_params = []
    _takes_rng = False

    def __init__(self, file_name, dir=None, file_type=None, key_split='.'):

        # First build full file_name
        self.file_name = file_name.strip()
        if dir:
            import os
            self.file_name = os.path.join(dir,self.file_name)
    
        if not file_type:
            import os
            name, ext = os.path.splitext(self.file_name)
            if ext.lower().startswith('.p'):
                file_type = 'PICKLE'
            elif ext.lower().startswith('.y'):
                file_type = 'YAML'
            elif ext.lower().startswith('.j'):
                file_type = 'JSON'
            else:
                raise ValueError('Unable to determine file_type from file_name ending')
        file_type = file_type.upper()
        if file_type not in ['PICKLE','YAML','JSON']:
            raise ValueError("file_type must be one of Pickle, YAML, or JSON if specified.")
        self.file_type = file_type

        self.key_split = key_split

        f = open(self.file_name)

        if file_type == 'PICKLE':
            import cPickle
            self.dict = cPickle.load(f)
        elif file_type == 'YAML':
            import yaml
            self.dict = yaml.load(f)
        else:
            import json
            self.dict = json.load(f)

        f.close()

    def get(self, key, default=None):
        # Make a list of keys according to our key_split parameter
        chain = key.split(self.key_split)
        d = self.dict
        while len(chain):
            k = chain.pop(0)
            
            # Try to convert to an integer:
            try: k = int(k)
            except ValueError: pass

            # If there are more keys, just set d to the next in the chanin.
            if chain: d = d[k]
            # Otherwise, return the result.
            else: 
                if k not in d and default is None:
                    raise ValueError("key=%s not found in dictionary"%key)
                return d.get(k,default)

        raise ValueError("Invalid key=%s given to Dict.get()"%key)

    # The rest of the functions are typical non-mutating functions for a dict, for which we just
    # pass the request along to self.dict.
    def __len__(self):
        return len(self.dict)

    def __getitem__(self, key):
        return self.dict[key]

    def __contains__(self, key):
        return key in self.dict

    def __iter__(self):
        return self.dict.__iter__

    def keys(self):
        return self.dict.keys()

    def values(self):
        return self.dict.values()

    def items(self):
        return self.dict.iteritems()

    def iterkeys(self):
        return self.dict.iterkeys()

    def itervalues(self):
        return self.dict.itervalues()

    def iteritems(self):
        return self.dict.iteritems()

