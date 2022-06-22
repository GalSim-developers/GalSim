Math
====

Nonlinear solver
----------------

.. doxygenclass:: galsim::Solve

Bessel and Related Functions
----------------------------

.. doxygenfunction:: galsim::math::cyl_bessel_j

.. doxygenfunction:: galsim::math::cyl_bessel_y

.. doxygenfunction:: galsim::math::cyl_bessel_k

.. doxygenfunction:: galsim::math::cyl_bessel_i

.. doxygenfunction:: galsim::math::j0

.. doxygenfunction:: galsim::math::j1

.. doxygenfunction:: galsim::math::getBesselRoot0

.. doxygenfunction:: galsim::math::getBesselRoot


Other mathematical functions
----------------------------

.. doxygenfunction:: galsim::math::sincos

.. doxygenfunction:: galsim::math::gamma_p

.. doxygenfunction:: galsim::math::sinc

.. doxygenfunction:: galsim::math::Si

.. doxygenfunction:: galsim::math::Ci


Horner's method for polynomial evaluation
-----------------------------------------

.. doxygenfunction:: galsim::math::Horner

.. doxygenfunction:: galsim::math::Horner2D

C++ Integration Functions
-------------------------

.. doxygenstruct:: galsim::integ::IntRegion

.. doxygenfunction:: galsim::integ::int1d(const UF&, typename UF::result_type, typename UF::result_type, const typename UF::result_type&, const typename UF::result_type&)

.. doxygenfunction:: galsim::integ::int1d(const UF&, IntRegion<typename UF::result_type>&, const typename UF::result_type&, const typename UF::result_type&)

.. doxygenfunction:: galsim::integ::int2d(const BF&, typename BF::result_type, typename BF::result_type, typename BF::result_type, typename BF::result_type, const typename BF::result_type&, const typename BF::result_type&)

.. doxygenfunction:: galsim::integ::int2d(const BF&, IntRegion<typename BF::result_type>&, const YREG&, const typename BF::result_type&, const typename BF::result_type&)

.. doxygenfunction:: galsim::integ::int2d(const BF&, IntRegion<typename BF::result_type>&, IntRegion<typename BF::result_type>&, const typename BF::result_type&, const typename BF::result_type&)

.. doxygenfunction:: galsim::integ::int3d(const TF&, typename TF::result_type, typename TF::result_type, typename TF::result_type, typename TF::result_type, typename TF::result_type, typename TF::result_type, const typename TF::result_type&, const typename TF::result_type&)

.. doxygenfunction:: galsim::integ::int3d(const TF&, IntRegion<typename TF::result_type>&, const YREG&, const ZREG&, const typename TF::result_type&, const typename TF::result_type&)

.. doxygenfunction:: galsim::integ::int3d(const TF&, IntRegion<typename TF::result_type>&, IntRegion<typename TF::result_type>&, IntRegion<typename TF::result_type>&, const typename TF::result_type&, const typename TF::result_type&)

.. doxygenfunction:: galsim::math::hankel_trunc

.. doxygenfunction:: galsim::math::hankel_inf

Misc Utilities
--------------

.. doxygenfunction:: galsim::math::isNan

.. doxygenfunction:: galsim::SetOMPThreads

.. doxygenfunction:: galsim::GetOMPThreads
