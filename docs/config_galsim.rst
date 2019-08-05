The galsim Executable
=====================

The normal way to run a GalSim simulation using a config file is ``galsim config.yaml``, where
``config.yaml`` is the name of the config file to be parsed.  For instance, to run demo1 (given
above), you would type::

    galsim demo1.yaml

Changing or adding parameters
-----------------------------

Sometimes it is convenient to be able to change some of the configuration parameters from the
command line, rather than edit the config file.  For instance, you might want to make a number
of simulations, which are nearly identical but differ in one or two specific attribute.

To enable this, you can provide the changed (or new) parameters on the command line after the 
name of the config file.  E.g.
to make several simulations that are identical except for the flux of the galaxy and the output
file, one could do::

    galsim demo1.yaml gal.flux=1.e4 output.file_name=demo1_1e4.fits
    galsim demo1.yaml gal.flux=2.e4 output.file_name=demo1_2e4.fits
    galsim demo1.yaml gal.flux=3.e4 output.file_name=demo1_3e4.fits
    galsim demo1.yaml gal.flux=4.e4 output.file_name=demo1_4e4.fits

Notice that the ``.`` is used to separate levels within the config hierarchy.
So ``gal.flux`` represents ``config['gal']['flux']``.

Splitting up a config job
-------------------------

For large simulations, one will typically want to split the job up into multiple smaller jobs,
each of which can be run on a single node or core.  The natural way to split this up is by
parceling some number of output files into each sub-job.  We make this splitting very easy using
the command line options ``-n`` and ``-j``.  The total number of jobs you want should be given
with ``-n``, and each separate job should be given a different ``-j``.  So to divide a run across
5 machines, you would run one of the following commands on each of the 5 different machines
(or more typically send these 5 commands as jobs in a queue system)::

    galsim config.yaml -n 5 -j 1
    galsim config.yaml -n 5 -j 2
    galsim config.yaml -n 5 -j 3
    galsim config.yaml -n 5 -j 4
    galsim config.yaml -n 5 -j 5

Other command line options
--------------------------

There are few other command line options that we describe here for completeness.

* ``-h`` or ``--help`` gives the help message.  This is really the definitive information about the
  ``galsim`` executable, so if that message disagrees with anything here, you should trust that
  information over what is written here.
* ``-v {0,1,2,3}`` or ``--verbosity {0,1,2,3}`` sets how verbose the logging output should be.
  The default is ``-v 1``, which provides some modest amount of output about each file being built.
  ``-v 2`` give more information about the progress within each output file, including one line of
  information about each object that is drawn.
  ``-v 3`` (debug mode) gives a lot of output and should be reserved for diagnosing runtime problems.
  ``-v 0`` turns off all logging output except for error messages.
* ``-l LOG_FILE`` or ``--log_file LOG_FILE`` gives a file name for writing the logging output.  If
  omitted, the default is to write to stdout.
* ``-f {yaml,json}`` or ``--file_type {yaml,json}`` defines what type of configuration file to parse.
  The default is to determine this from the file name extension, so it is not normally needed,
  but if you have non-standard file names, you might need to set this.
  * ``-m MODULE`` or ``--module MODULE`` gives a python module to import before parsing the config
  file.  This has been superseded by the ``modules`` top level field, which is normally more
  convenient.  However, this option is still allowed for backwards compatibility.
* ``-p`` or ``--profile`` turns on profiling information that gets output at the end of the run
  (or when multi-processing, at the end of execution of a process).  This can be useful for
  diagnosing where a simulation is spending most of its computation time.
* ``-n NJOBS`` or ``--njobs NJOBS`` sets the total number of jobs that this run is a part of. Used in conjunction with -j (--job).
* ``-j JOB`` or ``--job JOB`` sets the job number for this particular run. Must be in [1,njobs]. Used in conjunction with -n (--njobs).
* ``-x`` or ``--except_abort`` aborts the whole job whenever any file raises an exception rather than continuing on. (new in version 1.5)
* ``--version`` shows the version of GalSim.

