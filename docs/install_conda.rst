Installing With Conda
=====================

If you use conda (normally via the Anaconda Python distribution), then all of
the prerequisites and galsim itself are available from the conda-forge channel,
so you can use that as follows::

    conda install -c conda-forge galsim

If you want to install from source (e.g. to work on a development branch),
but use conda for the dependencies, you can do::

    git clone git@github.com:GalSim-developers/GalSim.git
    cd GalSim
    conda install astropy LSSTDESC.Coord
    pip install -e .

