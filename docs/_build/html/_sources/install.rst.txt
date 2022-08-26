Installation Instructions
#########################

GalSim is a Python module that has much of its implementation in C++ for
improved computational efficiency.
It is regularly tested on Python versions 3.7, 3.8, and 3.9 on both linux
and mac os.

It also seems to work on PyPy (both via conda-forge and the GitHub Actions
setup), although we don't consider this an officially supported system.
If you use GalSim with PyPy and experience any problems, we would appreciate
hearing about them.  Please open an issue describing any problems you find.

System requirements: GalSim currently only supports Linux and Mac OSX.
Possibly other POSIX-compliant systems, but we specifically do not
currently support Windows.

The usual way to install GalSim is now (starting with version 2.0) simply::

    pip install galsim

which will install the latest official release of GalSim.
For complete details, see `Installing With Pip`.

Another option If you use Anaconda Python is to use ``conda``::

    conda install -c conda-forge galsim

For more information, see `Installing With Conda`.


.. toctree::
    :maxdepth: 1

    install_pip.rst
    install_conda.rst

Running tests
=============

The simplest way to run our test suite by typing::

    python setup.py test

This should run all the Python-layer tests with pytest and also compile and
run the C++ test suite.

There are a number of packages that are used by the tests, but which are not
required for GalSim installation and running.  These should be installed
automatically by the above command, but you can install them manually via::

    pip install -r test_requirements.txt

(As usually, you may need to add either ``sudo`` or ``--user``.)

By default, the tests will run in parallel using the pytest plugins
``pytest-xdist`` and ``pytest-timeout`` (to manage how much time each test is
allowed to run).  If you want to run the Python tests in serial instead,
you can do this via::

    python setup.py test -j1

You can also use this to modify how many jobs will be spawned for running the
tests.

Or, you can run the Python tests yourself in the ``tests`` directory by typing::

    pytest test*.py

You can also run them with multiple jobs (e.g. for 4 jobs) by typing::

    pytest -n=4 --timeout=60 test*.py

You need the ``pytest-xdist`` and ``pytest-timeout`` plugins for this to work.

If you prefer to use nosetests, the equivalent command is::

    nosetests --processes=4 --process-timeout=60 test*.py

.. note::

    If your system does not have ``pytest`` installed, and you do not want
    to install it, you can run all the Python tests with the script ``run_all_tests``
    in the ``tests`` directory. If this finishes without an error, then all the tests
    have passed.  However, note that this script runs more tests than our normal
    test run using ``pytest``, so it may take quite a while to finish.  (The "all" in
    the file name means run **all** the tests including the slow ones that we normally
    skip.)



Running example scripts
=======================

The ``examples`` directory has a series of demo scripts::

    demo1.py, demo2.py, ...

These can be considered a tutorial on getting up to speed with GalSim. Reading
through these in order will introduce you to how to use most of the features of
GalSim in Python.  To run these scripts, type (e.g.)::

    python demo1.py

There are also a corresponding set of config files::

    demo1.yaml, demo2.yaml, ...

These files can be run using the executable ``galsim``, and will produce the
same output images as the Python scripts::

    galsim demo1.yaml

They are also well commented, and can be considered a parallel tutorial for
learning the config file usage of GalSim.

All demo scripts are designed to be run in the ``GalSim/examples`` directory.
Some of them access files in subdirectories of the ``examples`` directory, so they
would not work correctly from other locations.
