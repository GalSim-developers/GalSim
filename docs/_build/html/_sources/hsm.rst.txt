
The HSM Module
##############

Routines for adaptive moment estimation and PSF correction.

This module contains code for estimation of second moments of images, and for carrying out PSF
correction using a variety of algorithms.  The algorithms are described in
`Hirata & Seljak (2003) <http://adsabs.harvard.edu/abs/2003MNRAS.343..459H>`_, and were
tested/characterized using real data in
`Mandelbaum et al. (2005) <http://adsabs.harvard.edu/abs/2005MNRAS.361.1287M>`_.
Note that these routines for moment measurement and shear estimation are not accessible via config,
only via python.  There are a number of default settings for the code (often governing the tradeoff
between accuracy and speed) that can be adjusting using an optional ``hsmparams`` argument as
described below.

The moments that are estimated are "adaptive moments" (see the first paper cited above for details);
that is, they use an elliptical Gaussian weight that is matched to the image of the object being
measured.  The observed moments can be represented as a Gaussian sigma and a Shear object
representing the shape.

The PSF correction includes several algorithms, three that are re-implementations of methods
originated by others and one that was originated by Hirata & Seljak:

- One from `Kaiser, Squires, & Broadhurst (1995) <http://adsabs.harvard.edu/abs/1995ApJ...449..460K>`_, "KSB"

- One from `Bernstein & Jarvis (2002) <http://adsabs.harvard.edu/abs/2002AJ....123..583B>`_, "BJ"

- One that represents a modification by Hirata & Seljak (2003) of methods in Bernstein & Jarvis (2002), "LINEAR"

- One method from Hirata & Seljak (2003), "REGAUSS" (re-Gaussianization)

These methods return shear (or shape) estimators, which may not in fact satisfy conditions like
:math:`|e|<=1`, and so they are represented simply as e1/e2 or g1/g2 (depending on the method)
rather than using a Shear object, which IS required to satisfy :math:`|e|<=1`.

These methods are all based on correction of moments, but with different sets of assumptions.  For
more detailed discussion on all of these algorithms, see the relevant papers above.

Users can find a listing of the parameters that can be adjusted using the ``hsmparams`` keyword,
along with default values, under `galsim.hsm.HSMParams` below.


Shape Measurement Functions
===========================

.. autofunction:: galsim.hsm.FindAdaptiveMom

.. autofunction:: galsim.hsm.EstimateShear

HSM output
==========

.. autoclass:: galsim.hsm.ShapeData
    :members:

HSM parameters
==============

.. autoclass:: galsim.hsm.HSMParams
    :members:
