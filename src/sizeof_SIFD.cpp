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
/// @file sizeof_SIFD.cpp @brief Prints short, int, float and double sizes to stdout.
#include <cstdio>

int main(){

    printf("Sizeof short (bits) = %zu\n", sizeof(short) * 8);
    printf("Sizeof int (bits) = %zu\n", sizeof(int) * 8);
    printf("Sizeof long (bits) = %zu\n", sizeof(long) * 8);
    printf("Sizeof float (bits) = %zu\n", sizeof(float) * 8);
    printf("Sizeof double (bits) = %zu\n", sizeof(double) * 8);

}
