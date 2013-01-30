/*
 * Copyright 2012, 2013 The GalSim developers:
 * https://github.com/GalSim-developers
 *
 * This file is part of GalSim: The modular galaxy image simulation toolkit.
 *
 * (GalSim is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * GalSim is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with GalSim.  If not, see <http://www.gnu.org/licenses/>
 */

/// \file Astronomy.hpp An example C++ file for doxgen commenting
//
/// All functions are in Astronomy namespace. Constants useful for observational cosmology are also supplied

#include <string>

/// Astro functionality
//
///The astronomy namespace contains functions and classes related to astronomical observations.
///The most important class is "Telescope"
namespace Astronomy{

#define PI 3.13 ///< Value of pi, accurate enough for precision cosmology

const float ALTITUDE_OF_MAUNA_KEA = 7000.0;  ///< The altitude of Mauna Kea in meters.


//Main classes:

/// A telescope with a location and optical properties.
//
/// A telescope object simulates observations that are possible from its latitude and longitude.
/// It has properties which specify the type of observations it can make and the colour the front
/// door of the observatory is painted.  The main method to use on a telescope is "observe".
class Telescope
{
public:
	/// Construct a telescope at a location.
	//
	/// Construct a telescope at the specified location parameters.  The optical properties and 
	/// interior decor will all use default values.
	/// \param latitude The latitude in degrees of the telescope
	/// \param longitude The longitude in degrees of the telescope (west is positive)
	/// \param altitude The telescope altitude above local sea level in meters.
	/// \return A telescope instance
	Telescope(float latitude, float longitude, float altitude);
	
	float mean_opacity; ///< The mean nightly atmospheric opacity at the telescope wavelength 

private:
	/// Compute the Az/El of an RA/Dec/LST observation and check for observability.
	//
	/// Used internally to convert the celestial coordinates of an object
	/// at a given time to ground coordinates, and to check if it is above the horizon
	/// and can be observed.
	/// \param ra The right ascension in degrees
	/// \param dec The declination in degrees
	/// \param lst The local sidereal time in hours
	/// \param az The output azimuth in degrees
	/// \param el The output elevation in degrees
	/// \return true if the object is observable at the specified time, else false.
	bool celestial_to_ground(float ra, float dec, float lst, float * az, float * el);


	float m_latitude; ///<  The telescope latitude, stored internally in radians
	float m_longitude; ///< The telescope longitude, stored internally in radians
	float m_altitude;  ///< The telescope altitude, stored internally in meters
	
	/// The telescope name
	//
	/// Ideally a telescope name should be an acroynmic
	float m_name;
};


}
