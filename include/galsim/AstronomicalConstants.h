/* -*- c++ -*-
 * Copyright (c) 2012-2021 by the GalSim developers team on GitHub
 * https://github.com/GalSim-developers
 *
 * This file is part of GalSim: The modular galaxy image simulation toolkit.
 * https://github.com/GalSim-developers/GalSim
 *
 * GalSim is free software: redistribution and use in source and binary forms,
 * with or without modification, are permitted provided that the following
 * conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions, and the disclaimer given in the accompanying LICENSE
 *    file.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions, and the disclaimer given in the documentation
 *    and/or other materials provided with the distribution.
 */

// Include file with common "magic numbers" for Astronomy

#ifndef GalSim_AstroConst_H
#define GalSim_AstroConst_H

#ifndef PI
#define PI 3.14159265358979323
#endif

// conversion constants:
const double RadToArcsec=3600.*180./PI;
const double AU=1.49597870691e11;    //Astronomical Unit, m
const double Parsec=AU*RadToArcsec;
const double ABZeropoint=3.63e-23;         //AB=0 mag, in W/Hz/m^2
const double Jansky=1e-26;            //convert Jy to W/Hz/s/m^2
const double Planck=6.626068e-34;          //Planck's const in Js.
const double SpeedOfLight=2.99792458e8;    //speed of light in m/s
const double Boltzmann=1.38065e-23;    //Boltzmann's const, J/K
const double Micron=1e-6;     //micron, in m
const double HubbleLengthMpc=SpeedOfLight*1e-5; //c/H0, in h^-1 Mpc
const double HubbleDistance=HubbleLengthMpc*1e6*Parsec; //c/H0, in h^-1 m
const double RecombinationRedshift=1088.;

// Orbital Mechanics constants, units of AU, solar mass, and Year
const double  TPI             = 2.*PI;
const double  DEGREE          = PI/180.;               // One degree, in rad
const double  GM              = 4.*PI*PI/1.0000378;    //solar gravitation
const double  SolarSystemMass = 1.00134;
const double  ARCSEC          = PI/180./3600.;
const double  ARCMIN          = PI/180./60.;
const double  YEAR            = 1.;
const double  DAY             = 1./365.25;      //Julian day, 86400 s
const double  HOUR            = DAY/24.;
const double  MINUTE          = HOUR/60.;
const double  SECOND          = MINUTE/60.;
const double  SpeedOfLightAU  = 63241.06515;      //in AU/YR

//Obliquity of ecliptic at J2000
const double  EclipticInclination =23.43928*DEGREE;
const double  EclipticNode        =0.;
//Inclination and ascending node of invariable plane in the J2000
//equatorial system: ??? check this
const double  InvariableInclination=(23+22.11/3600.)*DEGREE;
const double  InvariableNode       = (3+(52.+23.7/60.)/60.)*DEGREE;

const double EarthMass        = 3.00349e-6; //Earth mass in Solar units
const double MJD0             = 2400000.5; //Offset for modified Julian dates


#endif

