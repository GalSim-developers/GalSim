Miscellaneous Utilities
=======================

We have a whole bunch of miscellaneous helper functions and classes in the ``galsim.utilities``
module.  Most of these are probably not particularly useful for anything other than internal
GalSim code.  But we highlight a few things that might be more widely useful beyond GalSim usage.

Decorators
----------

.. autoclass:: galsim.utilities.lazy_property

.. autoclass:: galsim.utilities.doc_inherit


LRU Cache
---------

.. autoclass:: galsim.utilities.LRU_Cache
    :members:


Other Possibly Useful Classes
-----------------------------

.. autoclass:: galsim.utilities.WeakMethod

.. autoclass:: galsim.utilities.OrderedWeakRef

.. autoclass:: galsim.utilities.SimpleGenerator

.. autoclass:: galsim.utilities.AttributeDict
    :members:
    :special-members:


Math Calculations
-----------------

.. autofunction:: galsim.utilities.horner

.. autofunction:: galsim.utilities.horner2d

.. autofunction:: galsim.utilities.binomial

.. autofunction:: galsim.utilities.nCr

.. autofunction:: galsim.utilities.rotate_xy

.. autofunction:: galsim.utilities.g1g2_to_e1e2


Utilities Related to NumPy Functions
------------------------------------

.. autofunction:: galsim.utilities.printoptions

.. autofunction:: galsim.utilities.roll2d

.. autofunction:: galsim.utilities.kxky

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

