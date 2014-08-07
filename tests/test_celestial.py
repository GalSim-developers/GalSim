# Copyright (c) 2003-2014 by Mike Jarvis
#
# TreeCorr is free software: redistribution and use in source and binary forms,
# with or without modification, are permitted provided that the following
# conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions, and the disclaimer given in the accompanying LICENSE
#    file.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions, and the disclaimer given in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the {organization} nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.


import numpy
import treecorr

# We'll use these a lot, so just import them.
from numpy import sin, cos, tan, arcsin, arccos, arctan, sqrt, pi

def test_angle_units():
    rad = treecorr.angle_units['rad']
    hour = treecorr.angle_units['hour']
    deg = treecorr.angle_units['deg']
    arcmin = treecorr.angle_units['arcmin']
    arcsec = treecorr.angle_units['arcsec']
    
    numpy.testing.assert_almost_equal(rad, 1.)
    numpy.testing.assert_almost_equal(hour, pi/12.)
    numpy.testing.assert_almost_equal(deg, pi/180.)
    numpy.testing.assert_almost_equal(arcmin, pi/(180.*60))
    numpy.testing.assert_almost_equal(arcsec, pi/(180.*3600))


def test_distance():

    # First, let's test some distances that are easy to figure out
    # without any spherical trig.
    eq1 = treecorr.CelestialCoord(0.,0.)  # point on the equator
    eq2 = treecorr.CelestialCoord(1.,0.)  # 1 radian along equator
    eq3 = treecorr.CelestialCoord(pi,0.) # antipode of eq1
    north_pole = treecorr.CelestialCoord(0.,pi/2.)  # north pole
    south_pole = treecorr.CelestialCoord(0.,-pi/2.) # south pole

    numpy.testing.assert_almost_equal(eq1.distanceTo(eq2), 1.)
    numpy.testing.assert_almost_equal(eq2.distanceTo(eq1), 1.)
    numpy.testing.assert_almost_equal(eq1.distanceTo(eq3), pi)
    numpy.testing.assert_almost_equal(eq2.distanceTo(eq3), pi-1.)

    numpy.testing.assert_almost_equal(north_pole.distanceTo(south_pole), pi)

    numpy.testing.assert_almost_equal(eq1.distanceTo(north_pole), pi/2.)
    numpy.testing.assert_almost_equal(eq2.distanceTo(north_pole), pi/2.)
    numpy.testing.assert_almost_equal(eq3.distanceTo(north_pole), pi/2.)
    numpy.testing.assert_almost_equal(eq1.distanceTo(south_pole), pi/2.)
    numpy.testing.assert_almost_equal(eq2.distanceTo(south_pole), pi/2.)
    numpy.testing.assert_almost_equal(eq3.distanceTo(south_pole), pi/2.)

    c1 = treecorr.CelestialCoord(0.234, 0.342)  # Some random point
    c2 = treecorr.CelestialCoord(0.234, -1.093) # Same meridian
    c3 = treecorr.CelestialCoord(pi + 0.234, -0.342) # Antipode
    c4 = treecorr.CelestialCoord(pi + 0.234, 0.832) # Different point on opposide meridian

    numpy.testing.assert_almost_equal(c1.distanceTo(c1), 0.)
    numpy.testing.assert_almost_equal(c1.distanceTo(c2), 1.435)
    numpy.testing.assert_almost_equal(c1.distanceTo(c3), pi)
    numpy.testing.assert_almost_equal(c1.distanceTo(c4), pi-1.174)

    # Now some that require spherical trig calculations. 
    # Importantly, this uses the more straightforward spherical trig formula, the cosine rule.
    # The CelestialCoord class uses a different formula that is more stable for very small
    # distances, which are typical in the correlation function calculation.
    c5 = treecorr.CelestialCoord(1.832, -0.723)  # Some other random point
    # The standard formula is:
    # cos(d) = sin(dec1) sin(dec2) + cos(dec1) cos(dec2) cos(delta ra)
    d = arccos(sin(0.342) * sin(-0.723) + cos(0.342) * cos(-0.723) * cos(1.832 - 0.234))
    numpy.testing.assert_almost_equal(c1.distanceTo(c5), d)

    # Tiny displacements should have dsq = (dra^2 cos^2 dec) + (ddec^2)
    c6 = treecorr.CelestialCoord(0.234 + 1.7e-9, 0.342)
    c7 = treecorr.CelestialCoord(0.234, 0.342 + 1.9e-9)
    c8 = treecorr.CelestialCoord(0.234 + 2.3e-9, 0.342 + 1.2e-9)

    # Note that the standard formula gets thsse wrong.  d comes back as 0.
    d = arccos(sin(0.342) * sin(0.342) + cos(0.342) * cos(0.342) * cos(1.7e-9))
    print 'd(c6) = ',1.7e-9 * cos(0.342), c1.distanceTo(c6), d
    d = arccos(sin(0.342) * sin(0.342+1.9e-9) + cos(0.342) * cos(0.342+1.9e-9) * cos(0.))
    print 'd(c7) = ',1.9e-9, c1.distanceTo(c7), d
    d = arccos(sin(0.342) * sin(0.342) + cos(0.342) * cos(0.342) * cos(1.2e-9))
    true_d = sqrt( (2.3e-9 * cos(0.342))**2 + 1.2e-9**2)
    print 'd(c7) = ',true_d, c1.distanceTo(c8), d
    numpy.testing.assert_almost_equal(c1.distanceTo(c6)/(1.7e-9 * cos(0.342)), 1.0)
    numpy.testing.assert_almost_equal(c1.distanceTo(c7)/1.9e-9, 1.0)
    numpy.testing.assert_almost_equal(c1.distanceTo(c8)/true_d, 1.0)


def test_angle():

    # Again, let's start with some answers we can get by inspection.
    eq1 = treecorr.CelestialCoord(0.,0.)  # point on the equator
    eq2 = treecorr.CelestialCoord(1.,0.)  # 1 radian along equator
    eq3 = treecorr.CelestialCoord(pi,0.) # antipode of eq1
    north_pole = treecorr.CelestialCoord(0.,pi/2.)  # north pole
    south_pole = treecorr.CelestialCoord(0.,-pi/2.) # south pole

    numpy.testing.assert_almost_equal(north_pole.angleBetween(eq1,eq2), 1.)
    numpy.testing.assert_almost_equal(north_pole.angleBetween(eq2,eq1), -1.)
    numpy.testing.assert_almost_equal(north_pole.angleBetween(eq2,eq3), pi-1.)
    numpy.testing.assert_almost_equal(north_pole.angleBetween(eq3,eq2), 1.-pi)
    numpy.testing.assert_almost_equal(south_pole.angleBetween(eq1,eq2), -1.)
    numpy.testing.assert_almost_equal(south_pole.angleBetween(eq2,eq1), 1.)
    numpy.testing.assert_almost_equal(south_pole.angleBetween(eq2,eq3), 1.-pi)
    numpy.testing.assert_almost_equal(south_pole.angleBetween(eq3,eq2), pi-1.)

    numpy.testing.assert_almost_equal(eq1.angleBetween(north_pole,eq2), -pi/2.)
    numpy.testing.assert_almost_equal(eq2.angleBetween(north_pole,eq1), pi/2.)

    numpy.testing.assert_almost_equal(north_pole.area(eq1,eq2), 1.)
    numpy.testing.assert_almost_equal(north_pole.area(eq2,eq1), 1.)
    numpy.testing.assert_almost_equal(south_pole.area(eq1,eq2), 1.)
    numpy.testing.assert_almost_equal(south_pole.area(eq2,eq1), 1.)

    # For arbitrary points, we can check that the spherical triangle satisfies
    # the spherical trig laws.
    cA = treecorr.CelestialCoord(0.234, 0.342)
    cB = treecorr.CelestialCoord(-0.193, 0.882)
    cC = treecorr.CelestialCoord(0.721, -0.561)

    a = cB.distanceTo(cC)
    b = cC.distanceTo(cA)
    c = cA.distanceTo(cB)
    A = cA.angleBetween(cB,cC)
    B = cB.angleBetween(cC,cA)
    C = cC.angleBetween(cA,cB)
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


def test_projection():

    # Test that a small triangle has the correct properties for each kind of projection
    center = treecorr.CelestialCoord(0.234,0.342)
    cA = treecorr.CelestialCoord(-0.193,0.882)
    cB = treecorr.CelestialCoord(-0.193 + 1.7e-6,0.882 + 1.2e-6)
    cC = treecorr.CelestialCoord(-0.193 - 2.4e-6,0.882 + 3.1e-6)

    a = cB.distanceTo(cC)
    b = cC.distanceTo(cA)
    c = cA.distanceTo(cB)
    A = cA.angleBetween(cB,cC)
    B = cB.angleBetween(cC,cA)
    C = cC.angleBetween(cA,cB)
    E = cA.area(cB,cC)

    #
    # The lambert is supposed to preserve area
    #

    pA = center.project(cA, projection='lambert')
    pB = center.project(cB, projection='lambert')
    pC = center.project(cC, projection='lambert')

    # The shoelace formula gives the area of a triangle given coordinates:
    # A = 1/2 abs( (x2-x1)*(y3-y1) - (x3-x1)*(y2-y1) )
    area = 0.5 * abs( (pB[0]-pA[0])*(pC[1]-pA[1]) - (pC[0]-pA[0])*(pB[1]-pA[1]) )
    print 'lambert area = ',area,E
    numpy.testing.assert_almost_equal(area / E, 1, decimal=5)

    # Check that project_rad does the same thing
    pA2 = center.project_rad(cA.ra, cA.dec, projection='lambert')
    numpy.testing.assert_array_almost_equal(pA, pA2)

    # Check the deprojection
    cA2 = center.deproject(*pA, projection='lambert')
    numpy.testing.assert_almost_equal(cA.ra, cA2.ra)
    numpy.testing.assert_almost_equal(cA.dec, cA2.dec)
    cA3 = center.deproject_rad(*pA, projection='lambert')
    numpy.testing.assert_array_almost_equal( [cA.ra, cA.dec], cA3 )
 
    # The angles are not preserved
    a = sqrt( (pB[0]-pC[0])**2 + (pB[1]-pC[1])**2 )
    b = sqrt( (pC[0]-pA[0])**2 + (pC[1]-pA[1])**2 )
    c = sqrt( (pA[0]-pB[0])**2 + (pA[1]-pB[1])**2 )
    cosA = ((pB[0]-pA[0])*(pC[0]-pA[0]) + (pB[1]-pA[1])*(pC[1]-pA[1])) / (b*c)
    cosB = ((pC[0]-pB[0])*(pA[0]-pB[0]) + (pC[1]-pB[1])*(pA[1]-pB[1])) / (c*a)
    cosC = ((pA[0]-pC[0])*(pB[0]-pC[0]) + (pA[1]-pC[1])*(pB[1]-pC[1])) / (a*b)

    print 'lambert cosA = ',cosA,cos(A)
    print 'lambert cosB = ',cosB,cos(B)
    print 'lambert cosC = ',cosC,cos(C)
 
    # The deproject jacobian should tell us how the area changes
    dudx, dudy, dvdx, dvdy = center.deproject_jac(*pA, projection='lambert')
    jac_area = abs(dudx*dvdy - dudy*dvdx)
    print 'lambert jac_area = ',jac_area,E/area
    numpy.testing.assert_almost_equal(jac_area, E/area, decimal=5)


    #
    # The stereographic is supposed to preserve angles
    #

    pA = center.project(cA, projection='stereographic')
    pB = center.project(cB, projection='stereographic')
    pC = center.project(cC, projection='stereographic')

    # The easiest way to compute the angles is from the dot products:
    # a.b = ab cos(C)
    a = sqrt( (pB[0]-pC[0])**2 + (pB[1]-pC[1])**2 )
    b = sqrt( (pC[0]-pA[0])**2 + (pC[1]-pA[1])**2 )
    c = sqrt( (pA[0]-pB[0])**2 + (pA[1]-pB[1])**2 )
    cosA = ((pB[0]-pA[0])*(pC[0]-pA[0]) + (pB[1]-pA[1])*(pC[1]-pA[1])) / (b*c)
    cosB = ((pC[0]-pB[0])*(pA[0]-pB[0]) + (pC[1]-pB[1])*(pA[1]-pB[1])) / (c*a)
    cosC = ((pA[0]-pC[0])*(pB[0]-pC[0]) + (pA[1]-pC[1])*(pB[1]-pC[1])) / (a*b)

    print 'stereographic cosA = ',cosA,cos(A)
    print 'stereographic cosB = ',cosB,cos(B)
    print 'stereographic cosC = ',cosC,cos(C)
    numpy.testing.assert_almost_equal(cosA,cos(A), decimal=5)
    numpy.testing.assert_almost_equal(cosB,cos(B), decimal=5)
    numpy.testing.assert_almost_equal(cosC,cos(C), decimal=5)
    
    # Check that project_rad does the same thing
    pA2 = center.project_rad(cA.ra, cA.dec, projection='stereographic')
    numpy.testing.assert_array_almost_equal(pA, pA2)

    # Check the deprojection
    cA2 = center.deproject(*pA, projection='stereographic')
    numpy.testing.assert_almost_equal(cA.ra, cA2.ra)
    numpy.testing.assert_almost_equal(cA.dec, cA2.dec)
    cA3 = center.deproject_rad(*pA, projection='stereographic')
    numpy.testing.assert_array_almost_equal( [cA.ra, cA.dec], cA3 )

    # The area is not preserved
    area = 0.5 * abs( (pB[0]-pA[0])*(pC[1]-pA[1]) - (pC[0]-pA[0])*(pB[1]-pA[1]) )
    print 'stereographic area = ',area,E
 
    # The deproject jacobian should tell us how the area changes
    dudx, dudy, dvdx, dvdy = center.deproject_jac(*pA, projection='stereographic')
    jac_area = abs(dudx*dvdy - dudy*dvdx)
    print 'stereographic jac_area = ',jac_area,E/area
    numpy.testing.assert_almost_equal(jac_area, E/area, decimal=5)


    #
    # The gnomonic is supposed to turn great circles into straight lines
    # I don't actually have any tests of that though...
    #

    pA = center.project(cA, projection='gnomonic')
    pB = center.project(cB, projection='gnomonic')
    pC = center.project(cC, projection='gnomonic')

    # Check that project_rad does the same thing
    pA2 = center.project_rad(cA.ra, cA.dec, projection='gnomonic')
    numpy.testing.assert_array_almost_equal(pA, pA2)

    # Check the deprojection
    cA2 = center.deproject(*pA, projection='gnomonic')
    numpy.testing.assert_almost_equal(cA.ra, cA2.ra)
    numpy.testing.assert_almost_equal(cA.dec, cA2.dec)
    cA3 = center.deproject_rad(*pA, projection='gnomonic')
    numpy.testing.assert_array_almost_equal( [cA.ra, cA.dec], cA3 )

    # The angles are not preserved
    a = sqrt( (pB[0]-pC[0])**2 + (pB[1]-pC[1])**2 )
    b = sqrt( (pC[0]-pA[0])**2 + (pC[1]-pA[1])**2 )
    c = sqrt( (pA[0]-pB[0])**2 + (pA[1]-pB[1])**2 )
    cosA = ((pB[0]-pA[0])*(pC[0]-pA[0]) + (pB[1]-pA[1])*(pC[1]-pA[1])) / (b*c)
    cosB = ((pC[0]-pB[0])*(pA[0]-pB[0]) + (pC[1]-pB[1])*(pA[1]-pB[1])) / (c*a)
    cosC = ((pA[0]-pC[0])*(pB[0]-pC[0]) + (pA[1]-pC[1])*(pB[1]-pC[1])) / (a*b)

    print 'gnomonic cosA = ',cosA,cos(A)
    print 'gnomonic cosB = ',cosB,cos(B)
    print 'gnomonic cosC = ',cosC,cos(C)
 
    # The area is not preserved
    area = 0.5 * abs( (pB[0]-pA[0])*(pC[1]-pA[1]) - (pC[0]-pA[0])*(pB[1]-pA[1]) )
    print 'gnomonic area = ',area,E

    # The deproject jacobian should tell us how the area changes
    dudx, dudy, dvdx, dvdy = center.deproject_jac(*pA, projection='gnomonic')
    jac_area = abs(dudx*dvdy - dudy*dvdx)
    print 'gnomonic jac_area = ',jac_area,E/area
    numpy.testing.assert_almost_equal(jac_area, E/area, decimal=5)


    #
    # The postel is supposed to preserve distance from the center
    #

    pA = center.project(cA, projection='postel')
    pB = center.project(cB, projection='postel')
    pC = center.project(cC, projection='postel')

    dA = sqrt( pA[0]**2 + pA[1]**2 )
    dB = sqrt( pB[0]**2 + pB[1]**2 )
    dC = sqrt( pC[0]**2 + pC[1]**2 )
    print 'postel dA = ',dA,center.distanceTo(cA)
    print 'postel dB = ',dB,center.distanceTo(cB)
    print 'postel dC = ',dC,center.distanceTo(cC)
    numpy.testing.assert_almost_equal( dA, center.distanceTo(cA) )
    numpy.testing.assert_almost_equal( dB, center.distanceTo(cB) )
    numpy.testing.assert_almost_equal( dC, center.distanceTo(cC) )

    # Check that project_rad does the same thing
    pA2 = center.project_rad(cA.ra, cA.dec, projection='postel')
    numpy.testing.assert_array_almost_equal(pA, pA2)

    # Check the deprojection
    cA2 = center.deproject(*pA, projection='postel')
    numpy.testing.assert_almost_equal(cA.ra, cA2.ra)
    numpy.testing.assert_almost_equal(cA.dec, cA2.dec)
    cA3 = center.deproject_rad(*pA, projection='postel')
    numpy.testing.assert_array_almost_equal( [cA.ra, cA.dec], cA3 )

    # The angles are not preserved
    a = sqrt( (pB[0]-pC[0])**2 + (pB[1]-pC[1])**2 )
    b = sqrt( (pC[0]-pA[0])**2 + (pC[1]-pA[1])**2 )
    c = sqrt( (pA[0]-pB[0])**2 + (pA[1]-pB[1])**2 )
    cosA = ((pB[0]-pA[0])*(pC[0]-pA[0]) + (pB[1]-pA[1])*(pC[1]-pA[1])) / (b*c)
    cosB = ((pC[0]-pB[0])*(pA[0]-pB[0]) + (pC[1]-pB[1])*(pA[1]-pB[1])) / (c*a)
    cosC = ((pA[0]-pC[0])*(pB[0]-pC[0]) + (pA[1]-pC[1])*(pB[1]-pC[1])) / (a*b)

    print 'postel cosA = ',cosA,cos(A)
    print 'postel cosB = ',cosB,cos(B)
    print 'postel cosC = ',cosC,cos(C)
 
    # The area is not preserved
    area = 0.5 * abs( (pB[0]-pA[0])*(pC[1]-pA[1]) - (pC[0]-pA[0])*(pB[1]-pA[1]) )
    print 'postel area = ',area,E

    # The deproject jacobian should tell us how the area changes
    dudx, dudy, dvdx, dvdy = center.deproject_jac(*pA, projection='postel')
    jac_area = abs(dudx*dvdy - dudy*dvdx)
    print 'postel jac_area = ',jac_area,E/area
    numpy.testing.assert_almost_equal(jac_area, E/area, decimal=5)


def test_precess():
    # I don't have much of a test here.  The formulae are what they are.
    # But it should at least be the case that a precession trip that ends up
    # back at the original epoch should leave the coord unchanged.
    orig = treecorr.CelestialCoord(0.234,0.342)

    c1 = orig.precess(2000., 1950.)
    c2 = c1.precess(1950., 1900.)
    c3 = c2.precess(1900., 2000.)
    numpy.testing.assert_almost_equal(c3.ra, orig.ra)
    numpy.testing.assert_almost_equal(c3.dec, orig.dec)

    # I found a website that does precession calculations, so check that we are 
    # consistent with them.
    # http://www.bbastrodesigns.com/coordErrors.html
    dra_1950 = -(2. + 39.07/60.)/60. * treecorr.angle_units['hour']
    ddec_1950 = -(16. + 16.3/60.)/60. * treecorr.angle_units['deg']
    print 'delta from website: ',dra_1950,ddec_1950
    print 'delta from precess: ',(c1.ra-orig.ra),(c1.dec-orig.dec)
    numpy.testing.assert_almost_equal(dra_1950, c1.ra-orig.ra, decimal=5)
    numpy.testing.assert_almost_equal(ddec_1950, c1.dec-orig.dec, decimal=5)

    dra_1900 = -(5. + 17.74/60.)/60. * treecorr.angle_units['hour']
    ddec_1900 = -(32. + 35.4/60.)/60. * treecorr.angle_units['deg']
    print 'delta from website: ',dra_1900,ddec_1900
    print 'delta from precess: ',(c2.ra-orig.ra),(c2.dec-orig.dec)
    numpy.testing.assert_almost_equal(dra_1900, c2.ra-orig.ra, decimal=5)
    numpy.testing.assert_almost_equal(ddec_1900, c2.dec-orig.dec, decimal=5)

def test_galactic():
    # According to wikipedia: http://en.wikipedia.org/wiki/Galactic_coordinate_system
    # the galactic center is located at 17h:45.6m, -28.94d
    center = treecorr.CelestialCoord( (17.+45.6/60.) * treecorr.angle_units['hour'],
                                      -28.94 * treecorr.angle_units['deg'] )
    print 'center.galactic = ',center.galactic()
    el,b = center.galactic()
    numpy.testing.assert_almost_equal(el, 0., decimal=3)
    numpy.testing.assert_almost_equal(b, 0., decimal=3)

    # The north pole is at 12h:51.4m, 27.13d
    north = treecorr.CelestialCoord( (12.+51.4/60.) * treecorr.angle_units['hour'],
                                     27.13 * treecorr.angle_units['deg'] )
    print 'north.galactic = ',north.galactic()
    el,b = north.galactic()
    numpy.testing.assert_almost_equal(b, pi/2., decimal=3)

    # The south pole is at 0h:51.4m, -27.13d
    south = treecorr.CelestialCoord( (0.+51.4/60.) * treecorr.angle_units['hour'],
                                     -27.13 * treecorr.angle_units['deg'] )
    print 'south.galactic = ',south.galactic()
    el,b = south.galactic()
    numpy.testing.assert_almost_equal(b, -pi/2., decimal=3)

    # The anti-center is at 5h:42.6m, 28.92d
    anticenter = treecorr.CelestialCoord( (5.+45.6/60.) * treecorr.angle_units['hour'],
                                          28.94 * treecorr.angle_units['deg'] )
    print 'anticenter.galactic = ',anticenter.galactic()
    el,b = anticenter.galactic()
    numpy.testing.assert_almost_equal(el, pi, decimal=3)
    numpy.testing.assert_almost_equal(b, 0., decimal=3)

if __name__ == '__main__':
    test_angle_units()
    test_distance()
    test_angle()
    test_projection()
    test_precess()
    test_galactic()
