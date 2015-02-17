# Copyright (c) 2012-2014 by the GalSim developers team on GitHub
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
"""@file catalog.py
Routines for controlling catalog input/output with GalSim. 
"""

import galsim
from galsim import pyfits
import numpy as np
import math
import os

bandpass = galsim.Bandpass(os.path.join(galsim.meta_data.share_dir, 'wfc_F814W.dat'),
                           wave_type='ang').thin().withZeropoint(25.94)

class Catalog(object):
    """A class storing the data from an input catalog.

    Each row corresponds to a different object to be built, and each column stores some item of
    information about that object (e.g. flux or half_light_radius).

    Initialization
    --------------

    @param file_name     Filename of the input catalog. (Required)
    @param dir           Optionally a directory name can be provided if `file_name` does not 
                         already include it.
    @param file_type     Either 'ASCII' or 'FITS'.  If None, infer from `file_name` ending.
                         [default: None]
    @param comments      The character used to indicate the start of a comment in an
                         ASCII catalog.  [default: '#']
    @param hdu           Which hdu to use for FITS files.  [default: 1]

    Attributes
    ----------

    After construction, the following attributes are available:

        nobjects   The number of objects in the catalog.
        ncols      The number of columns in the catalog.
        isfits     Whether the catalog is a fits catalog.
        names      For a fits catalog, the valid column names.

    """
    _req_params = { 'file_name' : str }
    _opt_params = { 'dir' : str , 'file_type' : str , 'comments' : str , 'hdu' : int }
    _single_params = []
    _takes_rng = False
    _takes_logger = False

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
            
    # When we make a proxy of this class (cf. galsim/config/stamp.py), the attributes
    # don't get proxied.  Only callable methods are.  So make method versions of these.
    def getNObjects(self) : return self.nobjects
    def isFits(self) : return self.isfits

    def read_ascii(self, comments, nobjects_only):
        """Read in an input catalog from an ASCII file.
        """
        # If all we care about is nobjects, this is quicker:
        if nobjects_only:
            # See the script devel/testlinecounting.py that tests several possibilities.
            # An even faster version using buffering is possible although it requires some care
            # around edge cases, so we use this one instead, which is "correct by inspection".
            with open(self.file_name) as f:
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
        from galsim import pyfits, pyfits_version
        raw_data = pyfits.getdata(self.file_name, hdu)
        if pyfits_version > '3.0':
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
        """Return the data for the given `index` and `col` in its native type.

        For ASCII catalogs, `col` is the column number.  
        For FITS catalogs, `col` is a string giving the name of the column in the FITS table.

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
        """Return the data for the given `index` and `col` as a float if possible
        """
        return float(self.get(index,col))

    def getInt(self, index, col):
        """Return the data for the given `index` and `col` as an int if possible
        """
        return int(self.get(index,col))


class Dict(object):
    """A class that reads a python dict from a file.

    After construction, it behaves like a regular python dict, with one exception.
    In order to facilitate getting values in a hierarchy of fields, we allow the '.'
    character to chain keys together for the get() method.  So,

        >>> d.get('noise.properties.variance')

    is expanded into

        >>> d['noise']['properties']['variance'] 

    Furthermore, if a "key" is really an integer, then it is used as such, which accesses 
    the corresponding element in a list.  e.g.

        >>> d.get('noise_models.2.variance')
        
    is equivalent to 

        >>> d['noise_models'][2]['variance']

    This makes it much easier to access arbitrary elements within parameter files.

    Caveat: The above prescription means that an element whose key really has a '.' in it
    won't be accessed correctly.  This is probably a rare occurrence, but the workaround is
    to set `key_split` to a different character or string and use that to chain the keys.


    @param file_name     Filename storing the dict.
    @param dir           Optionally a directory name can be provided if `file_name` does not 
                         already include it. [default: None]
    @param file_type     Options are 'Pickle', 'YAML', or 'JSON' or None.  If None, infer from 
                         `file_name` extension ('.p*', '.y*', '.j*' respectively).
                         [default: None]
    @param key_split     The character (or string) to use to split chained keys.  (cf. the 
                         description of this feature above.)  [default: '.']
    """
    _req_params = { 'file_name' : str }
    _opt_params = { 'dir' : str , 'file_type' : str, 'key_split' : str }
    _single_params = []
    _takes_rng = False
    _takes_logger = False

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

        with open(self.file_name) as f:

            if file_type == 'PICKLE':
                import cPickle
                self.dict = cPickle.load(f)
            elif file_type == 'YAML':
                import yaml
                self.dict = yaml.load(f)
            else:
                import json
                self.dict = json.load(f)


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

def makeCOSMOSCatalog(file_name, use_real=True, image_dir=None, dir=None, noise_dir=None,
                      preload=False, deep_sample=False, exclude_fail=True):
    """
    This routine makes a catalog of galaxies based on the COSMOS sample with F814W<23.5.

    Depending on the keyword arguments, particularly `use_real`, the catalog will either have
    information about real galaxies, or parametric ones.  To use this with either type of galaxies,
    you need to get the COSMOS datasets in the format that GalSim recognizes; see

        https://github.com/GalSim-developers/GalSim/wiki/RealGalaxy-Data

    option (1) for more information.  Note that if you want to make real galaxies you need to
    download the full tarball with all galaxy images, whereas if you want to make parametric
    galaxies you only need the supplemental catalogs, not the images.

    After getting the catalogs, there is a helper routine makeCOSMOSObj() that can make an object
    corresponding to any chosen galaxy in the catalog (whether real or parametric).  See
    help(galsim.makeCOSMOSObj) for more information.  As an interesting application and example of
    the usage of these routines, consider the following code:

        >>>> im_size = 64
        >>>> pix_scale = 0.05
        >>>> bandpass = galsim.Bandpass('wfc_F814W.dat',
                                        wave_type='ang').thin().withZeropoint(25.94)
        >>>> real_cat = galsim.makeCOSMOSCatalog('real_galaxy_catalog_23.5.fits',
                                                 dir='/path/to/COSMOS/data')
        >>>> param_cat = galsim.makeCOSMOSCatalog('real_galaxy_catalog_23.5_fits.fits',
                                                  dir='/path/to/COSMOS/data',
                                                  use_real=False, exclude_fail=False)
        >>>> psf = galsim.OpticalPSF(diam=2.4, lam=1000.) # bigger than HST F814W PSF.
        >>>> for ind in range(10):
        >>>>     real_gal = galsim.makeCOSMOSObj(real_cat, ind, pad_size=im_size*pix_scale)
        >>>>     param_gal = galsim.makeCOSMOSObj(param_cat, ind, chromatic=True)
        >>>>     real_obj = galsim.Convolve(real_gal, psf)
        >>>>     param_obj = galsim.Convolve(param_gal, psf)
        >>>>     im_real = galsim.Image(im_size, im_size)
        >>>>     im_param = galsim.Image(im_size, im_size)
        >>>>     try:
        >>>>         real_obj.drawImage(image=im_real, scale=pix_scale)
        >>>>         param_obj.drawImage(bandpass, image=im_param, scale=pix_scale)
        >>>>         im_real.write('im_real_'+str(ind)+'.fits')
        >>>>         im_param.write('im_param_'+str(ind)+'.fits')
        >>>>     except:
        >>>>         pass

    This code snippet will draw images of the first 10 objects in the COSMOS catalog, at slightly
    lower resolution than in COSMOS, with a real image and its parametric representation for each of
    those objects.

    @param file_name    The file containing the catalog.
    @param use_real     Use realistic galaxies or parametric ones?  [default: True]
    @param preload      Keyword that is only used for real galaxies, not parametric ones, to choose
                        whether to preload the header information.  If `preload=True`, the bulk of  
                        the I/O time is in the constructor.  If `preload=False`, there is
                        approximately the same total I/O time (assuming you eventually use most of
                        the image files referenced in the catalog), but it is spread over the
                        various calls to getGal() and getPSF().  [default: False]
    @param image_dir    Keyword that is only used for real galaxies, not parametric ones.
                        If a string containing no `/`, it is the relative path from the location of
                        the catalog file to the directory containing the galaxy/PDF images.
                        If a path (a string containing `/`), it is the full path to the directory
                        containing the galaxy/PDF images. [default: None]
    @param dir          The directory of catalog file. [default: None]
    @param noise_dir    Keyword that is only used for real galaxies, not parametric ones.
                        The directory of the noise files if different from the directory of the 
                        image files.  [default: image_dir]
    @param deep_sample  Modify fluxes and sizes of galaxies in order to roughly simulate an F814W<25
                        sample? [default: False]
    @param exclude_fail For catalogs of parametric galaxies, exclude those that have failures in the
                        parametric fits?  [default: True]

    @returns either a RealGalaxyCatalog or a pyfits.FITS_rec containing information about the real
    galaxies or parametric ones.
    """
    # Make fake deeper sample if necessary.
    if deep_sample:
        # Rescale the flux to get a limiting mag of 25 in F814W.  Current limiting mag is 23.5,
        # so it's a magnitude difference of 1.5.  Make the galaxies a factor of 0.6 smaller and
        # appropriately fainter.
        flux_factor = 10.**(-0.4*1.5)
        size_factor = 0.6
    else:
        flux_factor = 1.0
        size_factor = 1.0

    if use_real:
        # First, do the easy thing: real galaxies.  We make the galsim.RealGalaxyCatalog()
        # constructor do most of the work.
        cat = galsim.RealGalaxyCatalog(
            file_name, image_dir=image_dir, dir=dir, preload=preload, noise_dir=noise_dir)

        # We have a RealGalaxyCatalog object, can just attach stuff to it as new attributes and
        # return.
        cat.flux_factor = flux_factor
        cat.size_factor = size_factor
        return cat

    else:
        from real import parse_files_dirs

        # Find the file.
        use_file_name, _, _ = \
            parse_files_dirs(file_name, image_dir, dir, noise_dir)

        # Read in data.
        cat = pyfits.getdata(use_file_name)

        # Have to do some pyfits-related magic, then return the FITS_rec part of the FITS binary
        # table.
        col_list = [col for col in cat.columns]
        col_list.append(
            pyfits.Column(name='flux_factor', format='D',
                          array=flux_factor*np.ones(len(cat)))
            )
        col_list.append(
            pyfits.Column(name='size_factor', format='D',
                          array=size_factor*np.ones(len(cat)))
            )
        # The way to make the new BinTableHDU depends on the pyfits version.  Try the newer way
        # first:
        try:
            cat = pyfits.BinTableHDU.from_columns(pyfits.ColDefs(col_list))
        except:
            cat = pyfits.new_table(pyfits.ColDefs(col_list))

        # If requested, select galaxies based on existence of a usable fit.
        if exclude_fail:
            sersicfit_status = cat.data['fit_status'][:,4]
            bulgefit_status = cat.data['fit_status'][:,0]
            use_fit_ind = np.where(
                (sersicfit_status > 0) &
                (sersicfit_status < 5) &
                (bulgefit_status > 0) &
                (bulgefit_status < 5)
                )[0]
            return cat.data[use_fit_ind]
        else:
            return cat.data

def makeCOSMOSObj(cat, index, chromatic=False, pad_size=None):
    """
    Routine to take a catalog output by makeCOSMOSCatalog, and construct a GSObject corresponding to
    an object with a particular index.

    The fluxes are set such that drawing into an image with the COSMOS pixel scale should give the
    right pixel values to mimic the actual COSMOS image.

    @param cat        The output from makeCOSMOSCatalog
    @param index      The index of the object of interest in this catalog.
    @param chromatic  Make this a chromatic object, or not?  [default: False]
                      It is important to bear in mind that we do not actually have
                      spatially-resolved color information for these galaxies, so this keyword can
                      only be True if we are using parametric galaxies.  Even then, we simply do the
                      most arbitrary thing possible, which is to assign bulges an elliptical SED,
                      disks a disk-like SED, and Sersic galaxies with intermediate values of n some
                      intermediate SED.  We then normalize to give the right flux in F814W.
    @param pad_size   For realistic galaxies, the size of region requiring noise padding, in arcsec.

    @returns Either a GSObject or a chromatic object representing the galaxy of interest.
    """
    # Check whether this is a catalog entry for a real object or for a parametric one.
    if isinstance(cat, galsim.RealGalaxyCatalog):
        if pad_size is None:
            pad_size=0.25 # random and not completely ridiculous guess in arcsec
        if chromatic:
            raise RuntimeError("Cannot yet make real chromatic galaxies!")
        return _makeReal(cat, index, pad_size=pad_size)
    else:
        return _makeParam(cat, index, chromatic=chromatic)

def _makeReal(cat, index, pad_size):
    noise_pad_size = int(np.ceil(pad_size * np.sqrt(2.)))
    gal = galsim.RealGalaxy(cat, index=index, noise_pad_size=noise_pad_size)

    # Rescale the galaxy size.
    if hasattr(cat, 'size_factor'):
        gal.applyDilation(cat.size_factor)
    # Rescale the galaxy flux.
    if hasattr(cat, 'flux_factor'):
        gal *= cat.flux_factor
    return gal

def _makeParam(cat, index, chromatic=False):
    record = cat[index]

    if chromatic:
        # Read in some SEDs.
        sed_bulge = galsim.SED(os.path.join(galsim.meta_data.share_dir,'CWW_E_ext.sed'))
        sed_disk = galsim.SED(os.path.join(galsim.meta_data.share_dir,'CWW_Scd_ext.sed'))
        sed_intermed = galsim.SED(os.path.join(galsim.meta_data.share_dir,'CWW_Sbc_ext.sed'))

    # Get fit parameters.
    params = record.field('bulgefit')
    sparams = record.field('sersicfit')
    # Get the status flag for the fits.
    bstat = record.field('fit_status')[0]
    sstat = record.field('fit_status')[4]
    # Get the precomputed bulge-to-total ratio for the 2-component fits.
    dvc_btt = record.field('fit_dvc_btt')
    # Get the precomputed median absolute deviation for the 1- and 2-component fits.
    bmad = record.field('fit_mad_b')
    smad = record.field('fit_mad_s')

    # First decide if we can / should use bulgefit, otherwise sersicfit.  This decision process
    # depends on: the status flags for the fits, the bulge-to-total ratios (if near 0 or 1, just use
    # single component fits), the sizes for the bulge and disk (if <=0 then use single component
    # fits), the axis ratios for the bulge and disk (if <0.051 then use single component fits), and
    # a comparison of the median absolute deviations to see which is better.
    use_bulgefit = 1
    if bstat<1 or bstat>4 or dvc_btt<0.1 or dvc_btt>0.9 or np.isnan(dvc_btt) or params[9]<=0 or \
            params[1]<=0 or params[11]<0.051 or params[3]<0.051 or smad<bmad:
        use_bulgefit = 0
        # Then check if sersicfit is viable; if not, this object is a total failure:
        if sstat<1 or sstat>4 or sparams[1]<=0 or sparams[0]<=0:
            raise RuntimeError("Cannot make parametric model for this object")

    # Now, if we're supposed to use the 2-component fits, get all the parameters.
    if use_bulgefit:
        # Bulge parameters.
        bulge_q = params[11]
        bulge_beta = params[15]*galsim.radians
        bulge_hlr = 0.03*np.sqrt(bulge_q)*params[9]
        bulge_flux = 2.0*np.pi*3.607*(bulge_hlr**2)*params[8]/0.03**2
        # Disk parameters.
        disk_q = params[3]
        disk_beta = params[7]*galsim.radians
        disk_hlr = 0.03*np.sqrt(disk_q)*params[1]
        disk_flux = 2.0*np.pi*1.901*(disk_hlr**2)*params[0]/0.03**2
        bfrac = bulge_flux/(bulge_flux+disk_flux)
        # Make sure the bulge-to-total flux ratio is not nonsense.
        if bfrac < 0 or bfrac > 1 or np.isnan(bfrac):
            raise RuntimeError("Cannot make parametric model for this object")

        # Then make the object.
        if chromatic:
            bulge = galsim.DeVaucouleurs(flux=1., half_light_radius=record['size_factor']*bulge_hlr) \
                * sed_bulge.withMagnitude(
                record['mag_auto']-2.5*math.log10(bfrac*record['flux_factor']), bandpass)
            disk = galsim.Exponential(flux=1., half_light_radius=record['size_factor']*disk_hlr) \
                * sed_disk.withMagnitude(
                record['mag_auto']-2.5*math.log10((1.-bfrac)*record['flux_factor']), bandpass)
        else:
            bulge = galsim.DeVaucouleurs(flux = record['flux_factor']*bulge_flux,
                                         half_light_radius = record['size_factor']*bulge_hlr)
            disk = galsim.Exponential(flux = record['flux_factor']*disk_flux,
                                      half_light_radius = record['size_factor']*disk_hlr)
        # Apply shears for intrinsic shape.
        if bulge_q < 1.:
            bulge = bulge.shear(q=bulge_q, beta=bulge_beta)
        if disk_q < 1.:
            disk = disk.shear(q=disk_q, beta=disk_beta)
        return bulge+disk
    else:
        (fit_gal_flux, fit_gal_hlr, fit_gal_n, fit_gal_q, _, _, _, fit_gal_beta) = sparams

        gal_n = fit_gal_n
        # Fudge this if it is at the edge of the allowed n values.  Since GalSim (as of #325 and
        # #449) allow Sersic n in the range 0.3<=n<=6, the only problem is that the fits
        # occasionally go as low as n=0.2.
        if gal_n < 0.3: gal_n = 0.3
        gal_q = fit_gal_q
        gal_beta = fit_gal_beta*galsim.radians
        gal_hlr = 0.03*np.sqrt(gal_q)*fit_gal_hlr
        # Below is the calculation of the full Sersic n-dependent quantity that goes into the
        # conversion from surface brightness to flux, which here we're calling 'prefactor'.  In the
        # n=4 and n=1 cases above, this was precomputed, but here we have to calculate for each
        # value of n.
        tmp_ser = galsim.Sersic(gal_n, half_light_radius=1.)
        gal_bn = (1./tmp_ser.getScaleRadius())**(1./gal_n)
        prefactor = gal_n * _gammafn(2.*gal_n) * math.exp(gal_bn) / (gal_bn**(2.*gal_n))
        gal_flux = 2.*np.pi*prefactor*(gal_hlr**2)*fit_gal_flux/0.03**2

        if chromatic:
            gal = galsim.Sersic(gal_n, flux=1., half_light_radius=record['size_factor']*gal_hlr)
            if gal_n < 1.5:
                use_sed = sed_disk
            elif gal_n >= 1.5 and gal_n < 3.0:
                use_sed = sed_intermed
            else:
                use_sed = sed_bulge
            gal *= use_sed.withMagnitude(record['mag_auto']-2.5*math.log10(record['flux_factor']),
                                         bandpass)
        else:
            gal = galsim.Sersic(gal_n, flux=record['flux_factor']*gal_flux,
                                half_light_radius=record['size_factor']*gal_hlr)
        if gal_q < 1.:
            gal = gal.shear(q=gal_q, beta=gal_beta)
        return gal

def _gammafn(x):
    """The gamma function is present in python2.7's math module, but not 2.6.  So try using that,
    and if it fails, use some code from RosettaCode:
    http://rosettacode.org/wiki/Gamma_function#Python
    """
    try:
        import math
        return math.gamma(x)
    except:
        y  = float(x) - 1.0;
        sm = _gammafn._a[-1];
        for an in _gammafn._a[-2::-1]:
            sm = sm * y + an;
        return 1.0 / sm;

_gammafn._a = ( 1.00000000000000000000, 0.57721566490153286061, -0.65587807152025388108,
              -0.04200263503409523553, 0.16653861138229148950, -0.04219773455554433675,
              -0.00962197152787697356, 0.00721894324666309954, -0.00116516759185906511,
              -0.00021524167411495097, 0.00012805028238811619, -0.00002013485478078824,
              -0.00000125049348214267, 0.00000113302723198170, -0.00000020563384169776,
               0.00000000611609510448, 0.00000000500200764447, -0.00000000118127457049,
               0.00000000010434267117, 0.00000000000778226344, -0.00000000000369680562,
               0.00000000000051003703, -0.00000000000002058326, -0.00000000000000534812,
               0.00000000000000122678, -0.00000000000000011813, 0.00000000000000000119,
               0.00000000000000000141, -0.00000000000000000023, 0.00000000000000000002
             )

