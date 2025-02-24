Miscellaneous Utilities
=======================

We have a whole bunch of miscellaneous helper functions and classes in the ``galsim.utilities``
module.  Most of these are probably not particularly useful for anything other than internal
GalSim code.  But we highlight a few things that might be more widely useful beyond GalSim usage.

Decorators
----------

.. autoclass:: galsim.utilities.lazy_property

.. autoclass:: galsim.utilities.doc_inherit

.. autoclass:: galsim.utilities.timer


OpenMP Utilties
---------------

.. autofunction:: galsim.utilities.get_omp_threads

.. autofunction:: galsim.utilities.set_omp_threads

.. autoclass:: galsim.utilities.single_threaded


LRU Cache
---------

.. autoclass:: galsim.utilities.LRU_Cache
    :members:


Context Manager for writing AtmosphericScreen pickles
-----------------------------------------------------

.. autofunction:: galsim.utilities.pickle_shared


Other Possibly Useful Classes
-----------------------------

.. autoclass:: galsim.utilities.WeakMethod

.. autoclass:: galsim.utilities.OrderedWeakRef

.. autoclass:: galsim.utilities.SimpleGenerator


Math Calculations
-----------------

.. autofunction:: galsim.utilities.horner

.. autofunction:: galsim.utilities._horner

.. autofunction:: galsim.utilities.horner2d

.. autofunction:: galsim.utilities._horner2d

.. autofunction:: galsim.utilities.binomial

.. autofunction:: galsim.utilities.nCr

.. autofunction:: galsim.utilities.rotate_xy

.. autofunction:: galsim.utilities.g1g2_to_e1e2


Utilities Related to NumPy Functions
------------------------------------

.. autofunction:: galsim.utilities.printoptions

.. autofunction:: galsim.utilities.roll2d

.. autofunction:: galsim.utilities.kxky

.. autofunction:: galsim.utilities.merge_sorted


Test Suite Helper Functions and Contexts
----------------------------------------

.. autofunction:: galsim.utilities.check_pickle

.. autofunction:: galsim.utilities.check_all_diff

.. autoclass:: galsim.utilities.CaptureLog

.. autoclass:: galsim.utilities.Profile


Other Helper Functions
----------------------

.. autofunction:: galsim.utilities.isinteger

.. autofunction:: galsim.utilities.listify

.. autofunction:: galsim.utilities.dol_to_lod

.. autofunction:: galsim.utilities.functionize

.. autofunction:: galsim.utilities.ensure_dir


GalSim-specific Helper Functions
--------------------------------

.. autofunction:: galsim.utilities.interleaveImages

.. autofunction:: galsim.utilities.deInterleaveImage

.. autofunction:: galsim.utilities.thin_tabulated_values

.. autofunction:: galsim.utilities.old_thin_tabulated_values

.. autofunction:: galsim.utilities.parse_pos_args

.. autofunction:: galsim.utilities.rand_arr

.. autofunction:: galsim.utilities.convert_interpolant

.. autofunction:: galsim.utilities.structure_function

.. autofunction:: galsim.utilities.combine_wave_list

.. autofunction:: galsim.utilities.math_eval

.. autofunction:: galsim.utilities.unweighted_moments

.. autofunction:: galsim.utilities.unweighted_shape

.. autofunction:: galsim.utilities.rand_with_replacement

.. autofunction:: galsim.utilities.check_share_file

.. autofunction:: galsim.utilities.find_out_of_bounds_position

