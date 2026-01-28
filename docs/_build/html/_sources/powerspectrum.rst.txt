
Power Spectrum Shears
=====================

This is the "lensing engine" for calculating shears according to a Gaussian process with
specified E- and/or B-mode power spectra.

.. autoclass:: galsim.PowerSpectrum
    :members:

    .. automethod:: galsim.PowerSpectrum._getShear
    .. automethod:: galsim.PowerSpectrum._getConvergence
    .. automethod:: galsim.PowerSpectrum._getMagnification
    .. automethod:: galsim.PowerSpectrum._getLensing

.. autoclass:: galsim.lensing_ps.PowerSpectrumRealizer
    :members:

    .. automethod:: galsim.lensing_ps.PowerSpectrumRealizer.__call__

.. autofunction:: galsim.lensing_ps.theoryToObserved
