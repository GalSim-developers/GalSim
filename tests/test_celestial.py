# Copyright (c) 2012-2017 by the GalSim developers team on GitHub
# https://github.com/GalSim-developers
#
# This file is part of GalSim: The modular galaxy image simulation toolkit.
# https://github.com/GalSim-developers/GalSim
#
# GalSim is free software: redistribution and use in source and binary forms,
# with or without modification, are permitted provided that the following
# conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions, and the disclaimer given in the accompanying LICENSE
#    file.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions, and the disclaimer given in the documentation
#    and/or other materials provided with the distribution.
#

from __future__ import print_function
import numpy
import os
import sys

from galsim_test_helpers import *

imgdir = os.path.join(".", "SBProfile_comparison_images") # Directory containing the reference
                                                          # images.

try:
    import galsim
except ImportError:
    path, filename = os.path.split(__file__)
    sys.path.append(os.path.abspath(os.path.join(path, "..")))
    import galsim

# We'll use these a lot, so just import them.
from numpy import sin, cos, tan, arcsin, arccos, arctan, sqrt, pi


@timer
def test_angle():
    """Test basic construction and use of Angle and AngleUnit classes
    """
    # First Angle:
    theta1 = numpy.pi/4. * galsim.radians
    theta2 = 45 * galsim.degrees
    theta3 = 3 * galsim.hours
    theta4 = 45 * 60 * galsim.arcmin
    theta5 = galsim.Angle(45 * 3600 , galsim.arcsec) # Check explicit installation too.
    theta6 = galsim._Angle(numpy.pi/4.)  # Underscore constructor implicitly uses radians

    assert theta1.rad == numpy.pi/4.
    numpy.testing.assert_almost_equal(theta2.rad, numpy.pi/4., decimal=12)
    numpy.testing.assert_almost_equal(theta3.rad, numpy.pi/4., decimal=12)
    numpy.testing.assert_almost_equal(theta4.rad, numpy.pi/4., decimal=12)
    numpy.testing.assert_almost_equal(theta5.rad, numpy.pi/4., decimal=12)
    numpy.testing.assert_almost_equal(theta6.rad, numpy.pi/4., decimal=12)

    # Check wrapping
    theta6 = (45 + 360) * galsim.degrees
    assert abs(theta6.rad - theta1.rad) > 6.
    numpy.testing.assert_almost_equal(theta6.wrap().rad, theta1.rad, decimal=12)

    theta7 = (45 - 360) * galsim.degrees
    assert abs(theta7.rad - theta1.rad) > 6.
    numpy.testing.assert_almost_equal(theta7.wrap().rad, theta1.rad, decimal=12)

    # Check wrapping with non-default center
    pi_rad = pi * galsim.radians
    numpy.testing.assert_almost_equal(theta6.wrap(pi_rad).rad, theta1.rad, decimal=12)
    numpy.testing.assert_almost_equal(theta6.rad, theta1.wrap(2*pi_rad).rad, decimal=12)
    numpy.testing.assert_almost_equal(theta6.rad, theta1.wrap(3*pi_rad).rad, decimal=12)
    numpy.testing.assert_almost_equal(theta7.rad, theta1.wrap(-pi_rad).rad, decimal=12)
    numpy.testing.assert_almost_equal(theta7.rad, theta1.wrap(-2*pi_rad).rad, decimal=12)
    numpy.testing.assert_almost_equal(theta6.wrap(27*galsim.radians).rad,
                                      theta1.wrap(27*galsim.radians).rad, decimal=12)
    numpy.testing.assert_almost_equal(theta7.wrap(-127*galsim.radians).rad,
                                      theta1.wrap(-127*galsim.radians).rad, decimal=12)

    # Make a new AngleUnit as described in the AngleUnit docs
    gradians = galsim.AngleUnit(2. * numpy.pi / 400.)
    theta8 = 50 * gradians
    numpy.testing.assert_almost_equal(theta8.rad, numpy.pi/4., decimal=12)

    # Check simple math
    numpy.testing.assert_almost_equal((theta1 + theta2).rad, numpy.pi/2., decimal=12)
    numpy.testing.assert_almost_equal((4*theta3).rad, numpy.pi, decimal=12)
    numpy.testing.assert_almost_equal((4*theta4 - theta2).rad, 0.75 * numpy.pi, decimal=12)
    numpy.testing.assert_almost_equal((theta5/2.).rad, numpy.pi / 8., decimal=12)

    numpy.testing.assert_almost_equal(theta3 / galsim.radians, numpy.pi/4., decimal=12)
    numpy.testing.assert_almost_equal(theta1 / galsim.hours, 3., decimal=12)
    numpy.testing.assert_almost_equal(galsim.hours / galsim.arcmin, 15*60, decimal=12)

    # Check picklability
    do_pickle(galsim.radians)
    do_pickle(galsim.degrees)
    do_pickle(galsim.hours)
    do_pickle(galsim.arcmin)
    do_pickle(galsim.arcsec)
    do_pickle(gradians)
    do_pickle(theta1)
    do_pickle(theta2)
    do_pickle(theta3)
    do_pickle(theta4)
    do_pickle(theta5)
    do_pickle(theta6)
    do_pickle(theta7)
    do_pickle(theta8)


@timer
def test_celestialcoord_basic():
    """Basic tests of CelestialCoord construction. etc.
    """
    c1 = galsim.CelestialCoord(0. * galsim.radians, 0. * galsim.radians)
    numpy.testing.assert_almost_equal(c1.ra.rad, 0., decimal=12)
    numpy.testing.assert_almost_equal(c1.dec.rad, 0., decimal=12)

    c2 = galsim.CelestialCoord(11. * galsim.hours, -37. * galsim.degrees)
    numpy.testing.assert_almost_equal(c2.ra / galsim.hours, 11., decimal=12)
    numpy.testing.assert_almost_equal(c2.dec / galsim.degrees, -37., decimal=12)

    c3 = galsim.CelestialCoord(35. * galsim.hours, -37. * galsim.degrees)
    numpy.testing.assert_almost_equal(c3.ra / galsim.hours, 11., decimal=12)
    numpy.testing.assert_almost_equal(c3.dec / galsim.degrees, -37., decimal=12)

    c4 = galsim.CelestialCoord(-13. * galsim.hours, -37. * galsim.degrees)
    numpy.testing.assert_almost_equal(c4.ra / galsim.hours, 11., decimal=12)
    numpy.testing.assert_almost_equal(c4.dec / galsim.degrees, -37., decimal=12)

    numpy.testing.assert_almost_equal(c2.distanceTo(c3).rad, 0., decimal=12)
    numpy.testing.assert_almost_equal(c2.distanceTo(c4).rad, 0., decimal=12)

    x, y, z = c1.get_xyz()
    print('c1 is at x,y,z = ',x,y,z)
    np.testing.assert_equal((x,y,z), (1,0,0))
    assert c1 == galsim.CelestialCoord.from_xyz(x,y,z)

    x, y, z = c2.get_xyz()
    print('c2 is at x,y,z = ',x,y,z)
    assert c2 == galsim.CelestialCoord.from_xyz(x,y,z)

    # 0,0,0 shouldn't die at least, although the return value is a bit arbitrary.
    c5 = galsim.CelestialCoord.from_xyz(0,0,0)
    print('0,0,0 returns coord ',c5)

    # Check picklability
    do_pickle(c1)
    do_pickle(c2)
    do_pickle(c3)
    do_pickle(c4)
    do_pickle(c5)

    assert c1 == galsim.CelestialCoord(ra=0.*galsim.degrees, dec=0.*galsim.degrees)
    assert c2 == galsim.CelestialCoord(ra=165.*galsim.degrees, dec=-37.*galsim.degrees)
    assert c1 != c2
    assert c1 != c3
    assert c1 != c4
    # Depending on numerical rounding of the ra calculations, c2 may or may not come out
    # as equal to c3, c4, so don't check these pairings.


@timer
def test_celestialcoord_distance():
    """Test calculations of distances on the sphere."""
    # First, let's test some distances that are easy to figure out
    # without any spherical trig.
    eq1 = galsim.CelestialCoord(0. * galsim.radians, 0. * galsim.radians)  # point on the equator
    eq2 = galsim.CelestialCoord(1. * galsim.radians, 0. * galsim.radians)  # 1 radian along equator
    eq3 = galsim.CelestialCoord(pi * galsim.radians, 0. * galsim.radians) # antipode of eq1
    north_pole = galsim.CelestialCoord(0. * galsim.radians, pi/2. * galsim.radians)  # north pole
    south_pole = galsim.CelestialCoord(0. * galsim.radians, -pi/2. * galsim.radians) # south pole

    numpy.testing.assert_almost_equal(eq1.distanceTo(eq2).rad, 1.)
    numpy.testing.assert_almost_equal(eq2.distanceTo(eq1).rad, 1.)
    numpy.testing.assert_almost_equal(eq1.distanceTo(eq3).rad, pi)
    numpy.testing.assert_almost_equal(eq2.distanceTo(eq3).rad, pi-1.)

    numpy.testing.assert_almost_equal(north_pole.distanceTo(south_pole).rad, pi)

    numpy.testing.assert_almost_equal(eq1.distanceTo(north_pole).rad, pi/2.)
    numpy.testing.assert_almost_equal(eq2.distanceTo(north_pole).rad, pi/2.)
    numpy.testing.assert_almost_equal(eq3.distanceTo(north_pole).rad, pi/2.)
    numpy.testing.assert_almost_equal(eq1.distanceTo(south_pole).rad, pi/2.)
    numpy.testing.assert_almost_equal(eq2.distanceTo(south_pole).rad, pi/2.)
    numpy.testing.assert_almost_equal(eq3.distanceTo(south_pole).rad, pi/2.)

    # Some random point
    c1 = galsim.CelestialCoord(0.234 * galsim.radians, 0.342 * galsim.radians)
    # Same meridian
    c2 = galsim.CelestialCoord(0.234 * galsim.radians, -1.093 * galsim.radians)
    # Antipode
    c3 = galsim.CelestialCoord((pi + 0.234) * galsim.radians, -0.342 * galsim.radians)
    # Different point on opposide meridian
    c4 = galsim.CelestialCoord((pi + 0.234) * galsim.radians, 0.832 * galsim.radians)

    numpy.testing.assert_almost_equal(c1.distanceTo(c1).rad, 0.)
    numpy.testing.assert_almost_equal(c1.distanceTo(c2).rad, 1.435)
    numpy.testing.assert_almost_equal(c1.distanceTo(c3).rad, pi)
    numpy.testing.assert_almost_equal(c1.distanceTo(c4).rad, pi-1.174)

    # Now some that require spherical trig calculations.
    # Importantly, this uses the more straightforward spherical trig formula, the cosine rule.
    # The CelestialCoord class uses a different formula that is more stable for very small
    # distances, which are typical in the correlation function calculation.
    # Some other random point:
    c5 = galsim.CelestialCoord(1.832 * galsim.radians, -0.723 * galsim.radians)
    # The standard formula is:
    # cos(d) = sin(dec1) sin(dec2) + cos(dec1) cos(dec2) cos(delta ra)
    d = arccos(sin(0.342) * sin(-0.723) + cos(0.342) * cos(-0.723) * cos(1.832 - 0.234))
    numpy.testing.assert_almost_equal(c1.distanceTo(c5).rad, d)

    # Tiny displacements should have dsq = (dra^2 cos^2 dec) + (ddec^2)
    c6 = galsim.CelestialCoord((0.234 + 1.7e-9) * galsim.radians, 0.342 * galsim.radians)
    c7 = galsim.CelestialCoord(0.234 * galsim.radians, (0.342 + 1.9e-9) * galsim.radians)
    c8 = galsim.CelestialCoord((0.234 + 2.3e-9) * galsim.radians, (0.342 + 1.2e-9) * galsim.radians)

    # Note that the standard formula gets these wrong.  d comes back as 0.
    d = arccos(sin(0.342) * sin(0.342) + cos(0.342) * cos(0.342) * cos(1.7e-9))
    print('d(c6) = ',1.7e-9 * cos(0.342), c1.distanceTo(c6), d)
    d = arccos(sin(0.342) * sin(0.342+1.9e-9) + cos(0.342) * cos(0.342+1.9e-9) * cos(0.))
    print('d(c7) = ',1.9e-9, c1.distanceTo(c7), d)
    d = arccos(sin(0.342) * sin(0.342) + cos(0.342) * cos(0.342) * cos(1.2e-9))
    true_d = sqrt( (2.3e-9 * cos(0.342))**2 + 1.2e-9**2)
    print('d(c7) = ',true_d, c1.distanceTo(c8), d)
    numpy.testing.assert_almost_equal(c1.distanceTo(c6).rad/(1.7e-9 * cos(0.342)), 1.0)
    numpy.testing.assert_almost_equal(c1.distanceTo(c7).rad/1.9e-9, 1.0)
    numpy.testing.assert_almost_equal(c1.distanceTo(c8).rad/true_d, 1.0)


@timer
def test_celestialcoord_angleBetween():
    """Test calculations of angles between positions on the sphere."""
    # Again, let's start with some answers we can get by inspection.
    eq1 = galsim.CelestialCoord(0. * galsim.radians, 0. * galsim.radians)  # point on the equator
    eq2 = galsim.CelestialCoord(1. * galsim.radians, 0. * galsim.radians)  # 1 radian along equator
    eq3 = galsim.CelestialCoord(pi * galsim.radians, 0. * galsim.radians) # antipode of eq1
    north_pole = galsim.CelestialCoord(0. * galsim.radians, pi/2. * galsim.radians)  # north pole
    south_pole = galsim.CelestialCoord(0. * galsim.radians, -pi/2. * galsim.radians) # south pole

    numpy.testing.assert_almost_equal(north_pole.angleBetween(eq1,eq2).rad, -1.)
    numpy.testing.assert_almost_equal(north_pole.angleBetween(eq2,eq1).rad, 1.)
    numpy.testing.assert_almost_equal(north_pole.angleBetween(eq2,eq3).rad, 1.-pi)
    numpy.testing.assert_almost_equal(north_pole.angleBetween(eq3,eq2).rad, pi-1.)
    numpy.testing.assert_almost_equal(south_pole.angleBetween(eq1,eq2).rad, 1.)
    numpy.testing.assert_almost_equal(south_pole.angleBetween(eq2,eq1).rad, -1.)
    numpy.testing.assert_almost_equal(south_pole.angleBetween(eq2,eq3).rad, pi-1.)
    numpy.testing.assert_almost_equal(south_pole.angleBetween(eq3,eq2).rad, 1.-pi)

    numpy.testing.assert_almost_equal(eq1.angleBetween(north_pole,eq2).rad, pi/2.)
    numpy.testing.assert_almost_equal(eq2.angleBetween(north_pole,eq1).rad, -pi/2.)

    numpy.testing.assert_almost_equal(north_pole.area(eq1,eq2), 1.)
    numpy.testing.assert_almost_equal(north_pole.area(eq2,eq1), 1.)
    numpy.testing.assert_almost_equal(south_pole.area(eq1,eq2), 1.)
    numpy.testing.assert_almost_equal(south_pole.area(eq2,eq1), 1.)

    # For arbitrary points, we can check that the spherical triangle satisfies
    # the spherical trig laws.
    cA = galsim.CelestialCoord(0.234 * galsim.radians, 0.342 * galsim.radians)
    cB = galsim.CelestialCoord(-0.193 * galsim.radians, 0.882 * galsim.radians)
    cC = galsim.CelestialCoord(0.721 * galsim.radians, -0.561 * galsim.radians)

    a = cB.distanceTo(cC).rad
    b = cC.distanceTo(cA).rad
    c = cA.distanceTo(cB).rad
    A = cA.angleBetween(cB,cC).rad
    B = cB.angleBetween(cC,cA).rad
    C = cC.angleBetween(cA,cB).rad
    E = abs(A)+abs(B)+abs(C)-pi
    s = (a+b+c)/2.

    # Law of cosines:
    numpy.testing.assert_almost_equal(cos(c), cos(a)*cos(b) + sin(a)*sin(b)*cos(C))
    numpy.testing.assert_almost_equal(cos(a), cos(b)*cos(c) + sin(b)*sin(c)*cos(A))
    numpy.testing.assert_almost_equal(cos(b), cos(c)*cos(a) + sin(c)*sin(a)*cos(B))

    # Law of sines:
    numpy.testing.assert_almost_equal(sin(A) * sin(b), sin(B) * sin(a))
    numpy.testing.assert_almost_equal(sin(B) * sin(c), sin(C) * sin(b))
    numpy.testing.assert_almost_equal(sin(C) * sin(a), sin(A) * sin(c))

    # Alternate law of cosines:
    numpy.testing.assert_almost_equal(cos(C), -cos(A)*cos(B) + sin(A)*sin(B)*cos(c))
    numpy.testing.assert_almost_equal(cos(A), -cos(B)*cos(C) + sin(B)*sin(C)*cos(a))
    numpy.testing.assert_almost_equal(cos(B), -cos(C)*cos(A) + sin(C)*sin(A)*cos(b))

    # Spherical excess:
    numpy.testing.assert_almost_equal(cA.area(cB,cC), E)
    numpy.testing.assert_almost_equal(cA.area(cC,cB), E)
    numpy.testing.assert_almost_equal(cB.area(cA,cC), E)
    numpy.testing.assert_almost_equal(cB.area(cC,cA), E)
    numpy.testing.assert_almost_equal(cC.area(cB,cA), E)
    numpy.testing.assert_almost_equal(cC.area(cA,cB), E)

    # L'Huilier's formula for spherical excess:
    numpy.testing.assert_almost_equal(tan(E/4)**2, tan(s/2)*tan((s-a)/2)*tan((s-b)/2)*tan((s-c)/2))


@timer
def test_projection():
    """Test calculations of various projections."""
    # Test that a small triangle has the correct properties for each kind of projection
    center = galsim.CelestialCoord(0.234 * galsim.radians, 0.342 * galsim.radians)
    cA = galsim.CelestialCoord(-0.193 * galsim.radians, 0.882 * galsim.radians)
    cB = galsim.CelestialCoord((-0.193 + 1.7e-6) * galsim.radians,
                               (0.882 + 1.2e-6) * galsim.radians)
    cC = galsim.CelestialCoord((-0.193 - 2.4e-6) * galsim.radians,
                               (0.882 + 3.1e-6) * galsim.radians)

    a = cB.distanceTo(cC).rad
    b = cC.distanceTo(cA).rad
    c = cA.distanceTo(cB).rad
    A = cA.angleBetween(cB,cC).rad
    B = cB.angleBetween(cC,cA).rad
    C = cC.angleBetween(cA,cB).rad
    E = cA.area(cB,cC)

    #
    # The lambert is supposed to preserve area
    #

    # First the trivial case
    p0 = center.project(center, projection='lambert')
    assert p0 == galsim.PositionD(0,0)
    c0 = center.deproject(p0, projection='lambert')
    assert c0 == center
    np.testing.assert_almost_equal(center.deproject_jac(0,0, projection='lambert'), (1,0,0,1))

    pA = center.project(cA, projection='lambert')
    pB = center.project(cB, projection='lambert')
    pC = center.project(cC, projection='lambert')

    # The shoelace formula gives the area of a triangle given coordinates:
    # A = 1/2 abs( (x2-x1)*(y3-y1) - (x3-x1)*(y2-y1) )
    area = 0.5 * abs( (pB.x-pA.x)*(pC.y-pA.y) - (pC.x-pA.x)*(pB.y-pA.y) )
    area *= (galsim.arcsec / galsim.radians)**2
    print('lambert area = ',area,E)
    numpy.testing.assert_almost_equal(area / E, 1, decimal=5)

    # Check that project_rad does the same thing
    pA2 = center.project_rad(cA.ra.rad, cA.dec.rad, projection='lambert')
    numpy.testing.assert_array_almost_equal(pA.x, pA2[0])
    numpy.testing.assert_array_almost_equal(pA.y, pA2[1])

    # Check the deprojection
    cA2 = center.deproject(pA, projection='lambert')
    numpy.testing.assert_almost_equal(cA.ra.rad, cA2.ra.rad)
    numpy.testing.assert_almost_equal(cA.dec.rad, cA2.dec.rad)
    cA3 = center.deproject_rad(pA.x, pA.y, projection='lambert')
    numpy.testing.assert_array_almost_equal( [cA.ra.rad, cA.dec.rad], cA3 )

    # The angles are not preserved
    a = sqrt( (pB.x-pC.x)**2 + (pB.y-pC.y)**2 )
    b = sqrt( (pC.x-pA.x)**2 + (pC.y-pA.y)**2 )
    c = sqrt( (pA.x-pB.x)**2 + (pA.y-pB.y)**2 )
    cosA = ((pB.x-pA.x)*(pC.x-pA.x) + (pB.y-pA.y)*(pC.y-pA.y)) / (b*c)
    cosB = ((pC.x-pB.x)*(pA.x-pB.x) + (pC.y-pB.y)*(pA.y-pB.y)) / (c*a)
    cosC = ((pA.x-pC.x)*(pB.x-pC.x) + (pA.y-pC.y)*(pB.y-pC.y)) / (a*b)

    print('lambert cosA = ',cosA,cos(A))
    print('lambert cosB = ',cosB,cos(B))
    print('lambert cosC = ',cosC,cos(C))

    # The deproject jacobian should tell us how the area changes
    dudx, dudy, dvdx, dvdy = center.deproject_jac(pA.x, pA.y, projection='lambert')
    jac_area = abs(dudx*dvdy - dudy*dvdx)
    numpy.testing.assert_almost_equal(jac_area, E/area, decimal=5)


    #
    # The stereographic is supposed to preserve angles
    #

    # First the trivial case
    p0 = center.project(center, projection='stereographic')
    assert p0 == galsim.PositionD(0,0)
    c0 = center.deproject(p0, projection='stereographic')
    assert c0 == center
    np.testing.assert_almost_equal(center.deproject_jac(0,0, projection='stereographic'), (1,0,0,1))

    pA = center.project(cA, projection='stereographic')
    pB = center.project(cB, projection='stereographic')
    pC = center.project(cC, projection='stereographic')

    # The easiest way to compute the angles is from the dot products:
    # a.b = ab cos(C)
    a = sqrt( (pB.x-pC.x)**2 + (pB.y-pC.y)**2 )
    b = sqrt( (pC.x-pA.x)**2 + (pC.y-pA.y)**2 )
    c = sqrt( (pA.x-pB.x)**2 + (pA.y-pB.y)**2 )
    cosA = ((pB.x-pA.x)*(pC.x-pA.x) + (pB.y-pA.y)*(pC.y-pA.y)) / (b*c)
    cosB = ((pC.x-pB.x)*(pA.x-pB.x) + (pC.y-pB.y)*(pA.y-pB.y)) / (c*a)
    cosC = ((pA.x-pC.x)*(pB.x-pC.x) + (pA.y-pC.y)*(pB.y-pC.y)) / (a*b)

    print('stereographic cosA = ',cosA,cos(A))
    print('stereographic cosB = ',cosB,cos(B))
    print('stereographic cosC = ',cosC,cos(C))
    numpy.testing.assert_almost_equal(cosA,cos(A), decimal=5)
    numpy.testing.assert_almost_equal(cosB,cos(B), decimal=5)
    numpy.testing.assert_almost_equal(cosC,cos(C), decimal=5)

    # Check that project_rad does the same thing
    pA2 = center.project_rad(cA.ra.rad, cA.dec.rad, projection='stereographic')
    numpy.testing.assert_array_almost_equal(pA.x, pA2[0])
    numpy.testing.assert_array_almost_equal(pA.y, pA2[1])

    # Check the deprojection
    cA2 = center.deproject(pA, projection='stereographic')
    numpy.testing.assert_almost_equal(cA.ra.rad, cA2.ra.rad)
    numpy.testing.assert_almost_equal(cA.dec.rad, cA2.dec.rad)
    cA3 = center.deproject_rad(pA.x, pA.y, projection='stereographic')
    numpy.testing.assert_array_almost_equal( [cA.ra.rad, cA.dec.rad], cA3 )

    # The area is not preserved
    area = 0.5 * abs( (pB.x-pA.x)*(pC.y-pA.y) - (pC.x-pA.x)*(pB.y-pA.y) )
    area *= (galsim.arcsec / galsim.radians)**2
    print('stereographic area = ',area,E)

    # The deproject jacobian should tell us how the area changes
    dudx, dudy, dvdx, dvdy = center.deproject_jac(pA.x, pA.y, projection='stereographic')
    jac_area = abs(dudx*dvdy - dudy*dvdx)
    numpy.testing.assert_almost_equal(jac_area, E/area, decimal=5)


    #
    # The gnomonic is supposed to turn great circles into straight lines
    # I don't actually have any tests of that though...
    #

    # First the trivial case
    p0 = center.project(center, projection='gnomonic')
    assert p0 == galsim.PositionD(0,0)
    c0 = center.deproject(p0, projection='gnomonic')
    assert c0 == center
    np.testing.assert_almost_equal(center.deproject_jac(0,0, projection='gnomonic'), (1,0,0,1))

    pA = center.project(cA, projection='gnomonic')
    pB = center.project(cB, projection='gnomonic')
    pC = center.project(cC, projection='gnomonic')

    # Check that project_rad does the same thing
    pA2 = center.project_rad(cA.ra.rad, cA.dec.rad, projection='gnomonic')
    numpy.testing.assert_array_almost_equal(pA.x, pA2[0])
    numpy.testing.assert_array_almost_equal(pA.y, pA2[1])

    # Check the deprojection
    cA2 = center.deproject(pA, projection='gnomonic')
    numpy.testing.assert_almost_equal(cA.ra.rad, cA2.ra.rad)
    numpy.testing.assert_almost_equal(cA.dec.rad, cA2.dec.rad)
    cA3 = center.deproject_rad(pA.x, pA.y, projection='gnomonic')
    numpy.testing.assert_array_almost_equal( [cA.ra.rad, cA.dec.rad], cA3 )

    # The angles are not preserved
    a = sqrt( (pB.x-pC.x)**2 + (pB.y-pC.y)**2 )
    b = sqrt( (pC.x-pA.x)**2 + (pC.y-pA.y)**2 )
    c = sqrt( (pA.x-pB.x)**2 + (pA.y-pB.y)**2 )
    cosA = ((pB.x-pA.x)*(pC.x-pA.x) + (pB.y-pA.y)*(pC.y-pA.y)) / (b*c)
    cosB = ((pC.x-pB.x)*(pA.x-pB.x) + (pC.y-pB.y)*(pA.y-pB.y)) / (c*a)
    cosC = ((pA.x-pC.x)*(pB.x-pC.x) + (pA.y-pC.y)*(pB.y-pC.y)) / (a*b)

    print('gnomonic cosA = ',cosA,cos(A))
    print('gnomonic cosB = ',cosB,cos(B))
    print('gnomonic cosC = ',cosC,cos(C))

    # The area is not preserved
    area = 0.5 * abs( (pB.x-pA.x)*(pC.y-pA.y) - (pC.x-pA.x)*(pB.y-pA.y) )
    area *= (galsim.arcsec / galsim.radians)**2
    print('gnomonic area = ',area,E)

    # The deproject jacobian should tell us how the area changes
    dudx, dudy, dvdx, dvdy = center.deproject_jac(pA.x, pA.y, projection='gnomonic')
    jac_area = abs(dudx*dvdy - dudy*dvdx)
    numpy.testing.assert_almost_equal(jac_area, E/area, decimal=5)


    #
    # The postel is supposed to preserve distance from the center
    #

    # First the trivial case
    p0 = center.project(center, projection='postel')
    assert p0 == galsim.PositionD(0,0)
    c0 = center.deproject(p0, projection='postel')
    assert c0 == center
    np.testing.assert_almost_equal(center.deproject_jac(0,0, projection='postel'), (1,0,0,1))

    pA = center.project(cA, projection='postel')
    pB = center.project(cB, projection='postel')
    pC = center.project(cC, projection='postel')

    dA = sqrt( pA.x**2 + pA.y**2 )
    dB = sqrt( pB.x**2 + pB.y**2 )
    dC = sqrt( pC.x**2 + pC.y**2 )
    print('postel dA = ',dA,center.distanceTo(cA))
    print('postel dB = ',dB,center.distanceTo(cB))
    print('postel dC = ',dC,center.distanceTo(cC))
    numpy.testing.assert_almost_equal( dA, center.distanceTo(cA) / galsim.arcsec )
    numpy.testing.assert_almost_equal( dB, center.distanceTo(cB) / galsim.arcsec )
    numpy.testing.assert_almost_equal( dC, center.distanceTo(cC) / galsim.arcsec )

    # Check that project_rad does the same thing
    pA2 = center.project_rad(cA.ra.rad, cA.dec.rad, projection='postel')
    numpy.testing.assert_array_almost_equal(pA.x, pA2[0])
    numpy.testing.assert_array_almost_equal(pA.y, pA2[1])

    # Check the deprojection
    cA2 = center.deproject(pA, projection='postel')
    numpy.testing.assert_almost_equal(cA.ra.rad, cA2.ra.rad)
    numpy.testing.assert_almost_equal(cA.dec.rad, cA2.dec.rad)
    cA3 = center.deproject_rad(pA.x, pA.y, projection='postel')
    numpy.testing.assert_array_almost_equal( [cA.ra.rad, cA.dec.rad], cA3 )

    # The angles are not preserved
    a = sqrt( (pB.x-pC.x)**2 + (pB.y-pC.y)**2 )
    b = sqrt( (pC.x-pA.x)**2 + (pC.y-pA.y)**2 )
    c = sqrt( (pA.x-pB.x)**2 + (pA.y-pB.y)**2 )
    cosA = ((pB.x-pA.x)*(pC.x-pA.x) + (pB.y-pA.y)*(pC.y-pA.y)) / (b*c)
    cosB = ((pC.x-pB.x)*(pA.x-pB.x) + (pC.y-pB.y)*(pA.y-pB.y)) / (c*a)
    cosC = ((pA.x-pC.x)*(pB.x-pC.x) + (pA.y-pC.y)*(pB.y-pC.y)) / (a*b)

    print('postel cosA = ',cosA,cos(A))
    print('postel cosB = ',cosB,cos(B))
    print('postel cosC = ',cosC,cos(C))

    # The area is not preserved
    area = 0.5 * abs( (pB.x-pA.x)*(pC.y-pA.y) - (pC.x-pA.x)*(pB.y-pA.y) )
    area *= (galsim.arcsec / galsim.radians)**2
    print('postel area = ',area,E)

    # The deproject jacobian should tell us how the area changes
    dudx, dudy, dvdx, dvdy = center.deproject_jac(pA.x, pA.y, projection='postel')
    jac_area = abs(dudx*dvdy - dudy*dvdx)
    numpy.testing.assert_almost_equal(jac_area, E/area, decimal=5)


@timer
def test_precess():
    """Test precession between epochs."""
    # I don't have much of a test here.  The formulae are what they are.
    # But it should at least be the case that a precession trip that ends up
    # back at the original epoch should leave the coord unchanged.
    orig = galsim.CelestialCoord(0.234 * galsim.radians, 0.342 * galsim.radians)

    # First the trivial case of no precession.
    c0 = orig.precess(2000., 2000.)
    numpy.testing.assert_almost_equal(c0.ra.rad, orig.ra.rad)
    numpy.testing.assert_almost_equal(c0.dec.rad, orig.dec.rad)

    # Now to 1950 and back (via 1900).
    c1 = orig.precess(2000., 1950.)
    c2 = c1.precess(1950., 1900.)
    c3 = c2.precess(1900., 2000.)
    numpy.testing.assert_almost_equal(c3.ra.rad, orig.ra.rad)
    numpy.testing.assert_almost_equal(c3.dec.rad, orig.dec.rad)

    # I found a website that does precession calculations, so check that we are
    # consistent with them.
    # http://www.bbastrodesigns.com/coordErrors.html
    dra_1950 = -(2. + 39.07/60.)/60. * galsim.hours / galsim.radians
    ddec_1950 = -(16. + 16.3/60.)/60. * galsim.degrees / galsim.radians
    print('delta from website: ',dra_1950,ddec_1950)
    print('delta from precess: ',(c1.ra-orig.ra),(c1.dec-orig.dec))
    numpy.testing.assert_almost_equal(dra_1950, c1.ra.rad-orig.ra.rad, decimal=5)
    numpy.testing.assert_almost_equal(ddec_1950, c1.dec.rad-orig.dec.rad, decimal=5)

    dra_1900 = -(5. + 17.74/60.)/60. * galsim.hours / galsim.radians
    ddec_1900 = -(32. + 35.4/60.)/60. * galsim.degrees / galsim.radians
    print('delta from website: ',dra_1900,ddec_1900)
    print('delta from precess: ',(c2.ra-orig.ra),(c2.dec-orig.dec))
    numpy.testing.assert_almost_equal(dra_1900, c2.ra.rad-orig.ra.rad, decimal=5)
    numpy.testing.assert_almost_equal(ddec_1900, c2.dec.rad-orig.dec.rad, decimal=5)


@timer
def test_galactic():
    """Test the conversion from equatorial to galactic coordinates."""
    # According to wikipedia: http://en.wikipedia.org/wiki/Galactic_coordinate_system
    # the galactic center is located at 17h:45.6m, -28.94d
    # But I get more precise values from https://arxiv.org/pdf/1010.3773.pdf
    center = galsim.CelestialCoord(
        galsim.Angle.from_hms('17:45:37.1991'),
        galsim.Angle.from_dms('-28:56:10.2207'))
    print('center.galactic = ',center.galactic())
    el,b = center.galactic()
    np.testing.assert_almost_equal(el.wrap().rad, 0., decimal=8)
    np.testing.assert_almost_equal(b.rad, 0., decimal=8)

    # Go back from galactic coords to CelestialCoord
    center2 = galsim.CelestialCoord.from_galactic(el,b)
    np.testing.assert_allclose(center2.ra.rad, center.ra.rad)
    np.testing.assert_allclose(center2.dec.rad, center.dec.rad)

    # The north pole is at 12h:51.4m, 27.13d again with more precise values from the above paper.
    north = galsim.CelestialCoord(
        galsim.Angle.from_hms('12:51:26.27549'),
        galsim.Angle.from_dms('27:07:41.7043'))
    print('north.galactic = ',north.galactic())
    el,b = north.galactic()
    np.testing.assert_allclose(b.rad, pi/2.)
    north2 = galsim.CelestialCoord.from_galactic(el,b)
    np.testing.assert_allclose(north2.ra.rad, north.ra.rad)
    np.testing.assert_allclose(north2.dec.rad, north.dec.rad)

    south = galsim.CelestialCoord(
        galsim.Angle.from_hms('00:51:26.27549'),
        galsim.Angle.from_dms('-27:07:41.7043'))
    print('south.galactic = ',south.galactic())
    el,b = south.galactic()
    np.testing.assert_allclose(b.rad, -pi/2.)
    south2 = galsim.CelestialCoord.from_galactic(el,b)
    np.testing.assert_allclose(south2.ra.rad, south.ra.rad)
    np.testing.assert_allclose(south2.dec.rad, south.dec.rad)

    anticenter = galsim.CelestialCoord(
        galsim.Angle.from_hms('05:45:37.1991'),
        galsim.Angle.from_dms('28:56:10.2207'))
    print('anticenter.galactic = ',anticenter.galactic())
    el,b = anticenter.galactic()
    np.testing.assert_almost_equal(el.rad, pi, decimal=8)
    np.testing.assert_almost_equal(b.rad, 0., decimal=8)
    anticenter2 = galsim.CelestialCoord.from_galactic(el,b)
    np.testing.assert_allclose(anticenter2.ra.rad, anticenter.ra.rad)
    np.testing.assert_allclose(anticenter2.dec.rad, anticenter.dec.rad)


@timer
def test_ecliptic():
    """Test the conversion from equatorial to ecliptic coordinates."""
    # Use locations of ecliptic poles from http://en.wikipedia.org/wiki/Ecliptic_pole
    north_pole = galsim.CelestialCoord(
        galsim.Angle.from_hms('18:00:00.00'),
        galsim.Angle.from_dms('66:33:38.55'))
    el, b = north_pole.ecliptic()
    # North pole should have b=90 degrees, with el being completely arbitrary.
    numpy.testing.assert_almost_equal(b.rad, pi/2, decimal=6)

    south_pole = galsim.CelestialCoord(
        galsim.Angle.from_hms('06:00:00.00'),
        galsim.Angle.from_dms('-66:33:38.55'))
    el, b = south_pole.ecliptic()
    # South pole should have b=-90 degrees, with el being completely arbitrary.
    numpy.testing.assert_almost_equal(b.rad, -pi/2, decimal=6)

    # Also confirm that positions that should be the same in equatorial and ecliptic coordinates are
    # actually the same:
    vernal_equinox = galsim.CelestialCoord(0.*galsim.radians, 0.*galsim.radians)
    el, b = vernal_equinox.ecliptic()
    numpy.testing.assert_almost_equal(b.rad, 0., decimal=6)
    numpy.testing.assert_almost_equal(el.rad, 0., decimal=6)
    autumnal_equinox = galsim.CelestialCoord(pi*galsim.radians, 0.*galsim.radians)
    el, b = autumnal_equinox.ecliptic()
    numpy.testing.assert_almost_equal(el.rad, pi, decimal=6)
    numpy.testing.assert_almost_equal(b.rad, 0., decimal=6)

    # Finally, test the results of using a date to get ecliptic coordinates with respect to the sun,
    # instead of absolute ones. For this, use dates and times of vernal and autumnal equinox
    # in 2014 from
    # http://wwp.greenwichmeantime.com/longest-day/
    # and the conversion to Julian dates from
    # http://www.aavso.org/jd-calculator
    import datetime
    vernal_eq_date = datetime.datetime(2014,3,20,16,57,0)
    el, b = vernal_equinox.ecliptic(epoch=2014)
    el_rel, b_rel = vernal_equinox.ecliptic(epoch=2014, date=vernal_eq_date)
    # Vernal equinox: should have (el, b) = (el_rel, b_rel) = 0.0
    numpy.testing.assert_almost_equal(el_rel.rad, el.rad, decimal=3)
    numpy.testing.assert_almost_equal(b_rel.rad, b.rad, decimal=6)
    vernal2 = galsim.CelestialCoord.from_ecliptic(el_rel, b_rel, date=vernal_eq_date)
    np.testing.assert_almost_equal(vernal2.ra.wrap().rad, vernal_equinox.ra.rad, decimal=8)
    np.testing.assert_almost_equal(vernal2.dec.rad, vernal_equinox.dec.rad, decimal=8)

    # Now do the autumnal equinox: should have (el, b) = (pi, 0) = (el_rel, b_rel) when we look at
    # the time of the vernal equinox.
    el, b = autumnal_equinox.ecliptic(epoch=2014)
    el_rel, b_rel = autumnal_equinox.ecliptic(epoch=2014, date=vernal_eq_date)
    numpy.testing.assert_almost_equal(el_rel.rad, el.rad, decimal=3)
    numpy.testing.assert_almost_equal(b_rel.rad, b.rad, decimal=6)
    autumnal2 = galsim.CelestialCoord.from_ecliptic(el_rel, b_rel, date=vernal_eq_date)
    np.testing.assert_almost_equal(autumnal2.ra.wrap(pi*galsim.radians).rad,
                                   autumnal_equinox.ra.wrap(pi*galsim.radians).rad, decimal=8)
    np.testing.assert_almost_equal(autumnal2.dec.rad, autumnal_equinox.dec.rad, decimal=8)

    # And check that if it's the date of the autumnal equinox (sun at (180, 0)) but we're looking at
    # the position of the vernal equinox (0, 0), then (el_rel, b_rel) = (-180, 0)
    autumnal_eq_date = datetime.datetime(2014,9,23,2,29,0)
    el_rel, b_rel = vernal_equinox.ecliptic(epoch=2014, date=autumnal_eq_date)
    numpy.testing.assert_almost_equal(el_rel.rad, -pi, decimal=3)
    numpy.testing.assert_almost_equal(b_rel.rad, 0., decimal=6)

    # And check that if it's the date of the vernal equinox (sun at (0, 0)) but we're looking at
    # the position of the autumnal equinox (180, 0), then (el_rel, b_rel) = (180, 0)
    el_rel, b_rel = autumnal_equinox.ecliptic(epoch=2014, date=vernal_eq_date)
    numpy.testing.assert_almost_equal(el_rel.rad, pi, decimal=3)
    numpy.testing.assert_almost_equal(b_rel.rad, 0., decimal=6)

    # Check round-trips: go from CelestialCoord to ecliptic back to equatorial, and make sure
    # results are the same.
    north_pole_2 = galsim.CelestialCoord.from_ecliptic(*north_pole.ecliptic(epoch=2014), epoch=2014)
    numpy.testing.assert_almost_equal(north_pole.ra.rad, north_pole_2.ra.rad, decimal=6)
    numpy.testing.assert_almost_equal(north_pole.dec.rad, north_pole_2.dec.rad, decimal=6)
    south_pole_2 = galsim.CelestialCoord.from_ecliptic(*south_pole.ecliptic(epoch=2014), epoch=2014)
    numpy.testing.assert_almost_equal(south_pole.ra.rad, south_pole_2.ra.rad, decimal=6)
    numpy.testing.assert_almost_equal(south_pole.dec.rad, south_pole_2.dec.rad, decimal=6)
    vernal_equinox_2 = galsim.CelestialCoord.from_ecliptic(*vernal_equinox.ecliptic(epoch=2014), epoch=2014)
    numpy.testing.assert_almost_equal(vernal_equinox.ra.rad, vernal_equinox_2.ra.rad, decimal=6)
    numpy.testing.assert_almost_equal(vernal_equinox.dec.rad, vernal_equinox_2.dec.rad,
                                      decimal=6)
    autumnal_equinox_2 = galsim.CelestialCoord.from_ecliptic(*autumnal_equinox.ecliptic(epoch=2014), epoch=2014)
    numpy.testing.assert_almost_equal(autumnal_equinox.ra.rad, autumnal_equinox_2.ra.rad,
                                      decimal=6)
    numpy.testing.assert_almost_equal(autumnal_equinox.dec.rad, autumnal_equinox_2.dec.rad,
                                      decimal=6)


if __name__ == '__main__':
    test_angle()
    test_celestialcoord_basic()
    test_celestialcoord_distance()
    test_celestialcoord_angleBetween()
    test_projection()
    test_precess()
    test_galactic()
    test_ecliptic()
