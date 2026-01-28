"Real" Galaxies
===============

Individual Real Galaxies
------------------------

The `RealGalaxy` class uses images of galaxies from real astrophysical data (e.g. the Hubble Space
Telescope), along with a PSF model of the optical properties of the telescope that took these
images, to simulate new galaxy images with a different (must be larger) telescope PSF.  A
description of the simulation method can be found in Section 5 of Mandelbaum et al. (2012; MNRAS,
540, 1518), although note that the details of the implementation in Section 7 of that work are not
relevant to the more recent software used here.

The `RealGalaxyCatalog` class stores all required information about a real galaxy simulation
training sample and accompanying PSF model.
This modelling requires external data for the galaxy images and PSF models, which is read into a
`RealGalaxyCatalog` object from FITS files. An example catalog of 100 real galaxies is in the
repository itself at

    GalSim/examples/data/real_galaxy_catalog_23.5_example.fits

For access to larger catalogs of objects, see `Downloading the COSMOS Catalog` below.

.. autoclass:: galsim.RealGalaxy
    :members:
    :show-inheritance:

.. autoclass:: galsim.RealGalaxyCatalog
    :members:

Realistic Scene
---------------

The `COSMOSCatalog` class is also based on the above `RealGalaxyCatalog`, and has functionality
for defining a "sky scene", i.e., a galaxy sample with reasonable properties that can then be
placed throughout a large image.  It can simulate either `RealGalaxy` objects using the HST images
or parametric models based on those images.

.. note::
   Currently, this only includes routines for making a COSMOS-based galaxy sample, but it could be
   expanded to include star samples as well.


.. autoclass:: galsim.COSMOSCatalog
    :members:

.. autoclass:: galsim.GalaxySample
    :members:

Downloading the COSMOS Catalog
------------------------------

A set of ~56 000 real galaxy images with I<23.5, or another set of ~87 000 with I<25.2, with
original PSFs, can be downloaded from Zenodo:

https://zenodo.org/record/3242143

The tar ball for the I<23.5 sample (``COSMOS_23.5_training_sample.tar.gz``) is roughly 4 GB and
contains catalogs and images with a README. The tar ball for the I<25.2 sample
(``COSMOS_25.2_training_sample.tar.gz``) is of similar size and format.

GalSim also comes with a script ``galsim_download_cosmos`` that downloads the I<23.5 sample.
It works with both samples, with the I<25.2 sample being the default but with keyword arguments
to choose between the two::

    usage: galsim_download_cosmos [-h] [-v {0,1,2,3}] [-f] [-q] [-u] [--save]
                                  [-d DIR] [-s {23.5,25.2}] [--nolink]

    This program will download the COSMOS RealGalaxy catalog and images and place
    them in the GalSim share directory so they can be used as the default files
    for the RealGalaxyCatalog class. See https://github.com/GalSim-
    developers/GalSim/wiki/RealGalaxy%20Data for more details about the files
    being downloaded.

    optional arguments:
      -h, --help            show this help message and exit
      -v {0,1,2,3}, --verbosity {0,1,2,3}
                            Integer verbosity level: min=0, max=3 [default=2]
      -f, --force           Force overwriting the current file if one exists
      -q, --quiet           Don't ask about re-downloading an existing file.
                            (implied by verbosity=0)
      -u, --unpack          Re-unpack the tar file if not downloading
      --save                Save the tarball after unpacking.
      -d DIR, --dir DIR     Install into an alternate directory and link from the
                            share/galsim directory
      -s {23.5,25.2}, --sample {23.5,25.2}
                            Flux limit for sample to download; either 23.5 or 25.2
      --nolink              Don't link to the alternate directory from
                            share/galsim

    Note: The unpacked files total almost 6 GB in size!

.. note::

    The ``galsim_download_cosmos`` program will put the downloaded files into a subdirectory
    of the ``galsim.meta_data.share_dir`` directory.  (cf. `Shared Data`)
    This is normally convenient for access, since classes such as `RealGalaxyCatalog` and
    `COSMOSCatalog` will look in this directory automatically for you.  However, if you
    reinstall GalSim, everything in this directory will be removed and overwritten.
    Therefore, we normally recommend using the ``-d DIR`` option to place the downloaded
    files into another location.  E.g.::

        galsim_download_cosmos -d ~/share

    It will still be required to rerun this after reinstalling GalSim, but it will notice that
    you already have the files downloaded and merely update the symbolic link.

Instructions for how to download a copy of the GREAT3 data are found at

https://github.com/barnabytprowe/great3-public#how-to-get-the-data

HSC Postage Stamp Data
----------------------

The HST postage stamp data from

http://adsabs.harvard.edu/abs/2017arXiv171000885M

which includes studies of the impact of blending on shear estimation in HSC, was released as part
of the HSC survey's second incremental data release. The sample is larger than the above, goes to
the depth of COSMOS, and does not have nearby objects masked.

The links to download it and the instructions on its use are at

https://hsc-release.mtk.nao.ac.jp/doc/index.php/weak-lensing-simulation-catalog-pdr1/

