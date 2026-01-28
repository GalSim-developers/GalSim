Installing With Conda
=====================

If you use conda (normally via the Anaconda Python distribution), then all of
the prerequisites and galsim itself are available from the conda-forge channel,
so you can use that as follows::

    conda install -c conda-forge galsim

Also, if you prefer to use the defaults channel, then (at least as of this
writing), it had all the items in conda_requirements.txt, except for pybind11.
So if you have conda-forge in your list of channels, but it comes after
defaults, then that should still work and pybind11 will be the only one that
will need the conda-forge channel.

If you want to install from source (e.g. to work on a development branch),
but use conda for the dependencies, you can do::

    git clone git@github.com:GalSim-developers/GalSim.git
    cd GalSim
    conda install --file conda_requirements.txt
    python setup.py install

