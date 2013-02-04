// -*- c++ -*-
/*
 * Copyright 2012, 2013 The GalSim developers:
 * https://github.com/GalSim-developers
 *
 * This file is part of GalSim: The modular galaxy image simulation toolkit.
 *
 * GalSim is free software: you can redistribute it and/or modify
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
/// @file sizeof_SIFD.cpp @brief Prints short, int, float and double sizes to stdout. 
#include <cstdio>

int main(){

    printf("Sizeof short (bits) = %zu\n", sizeof(short) * 8);
    printf("Sizeof int (bits) = %zu\n", sizeof(int) * 8);
    printf("Sizeof long (bits) = %zu\n", sizeof(long) * 8);
    printf("Sizeof float (bits) = %zu\n", sizeof(float) * 8);
    printf("Sizeof double (bits) = %zu\n", sizeof(double) * 8);

}
