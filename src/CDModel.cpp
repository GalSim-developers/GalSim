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

#include "CDModel.h"

namespace galsim {

    template <typename T>
    void ApplyCD(ImageView<T>& output, const BaseImage<T>& input,
                 const BaseImage<double>& aL, const BaseImage<double>& aR,
                 const BaseImage<double>& aB, const BaseImage<double>& aT,
                 const int dmax, const double gain_ratio)
    {
        // images aL, aR, aB, aT contain shift coefficients for left, right, bottom and top border
        // dmax is maximum separation considered
        // gain_ratio is gain_image/gain_flat when shift coefficients were measured on flat and
        // image has different gain

        // Perform sanity check
        if(dmax < 0) throw ImageError("Attempt to apply CD model with invalid extent");

        // compare eqn. 4.5 in Antilogus+2014
        // output is
        //        (1)   input +
        //        (2)   interpolated version of image at pixel borders *
        //        (3)   image convolved with shift coefficients

        for(int x=input.getXMin(); x<=input.getXMax(); x++){
            for(int y=input.getYMin(); y<=input.getYMax(); y++){

                // (1) input image
                double f = input(x, y);

                // (2) interpolated version of image at pixel borders
                double fT = 0.; if(y < input.getYMax()) fT = (f + input(x, y + 1)) / 2.;
                double fB = 0.; if(y > input.getYMin()) fB = (f + input(x, y - 1)) / 2.;
                double fR = 0.; if(x < input.getXMax()) fR = (f + input(x + 1, y)) / 2.;
                double fL = 0.; if(x > input.getXMin()) fL = (f + input(x - 1, y)) / 2.;

                // (3) convolution of image with shift coefficient matrix
                for(int iy=-dmax; iy<=dmax; iy++){
                    for(int ix=-dmax; ix<=dmax; ix++){

                        if(x+ix<input.getXMin() || x+ix>input.getXMax() ||
                           y+iy<input.getYMin() || y+iy>input.getYMax()) {
                            continue; // a non-existent pixel is not going to move us
                        }
                        double qkl = input(x + ix, y + iy) * gain_ratio;

                        if(y + 1 - iy >= input.getYMin() && y + 1 - iy <= input.getYMax())
                            // don't apply shift if pixel mirrored at t border non-existent
                            f += qkl * fT * aT(ix+dmax+1, iy+dmax+1);

                        if(y - 1 - iy >= input.getYMin() && y - 1 - iy <= input.getYMax())
                            // don't apply shift if pixel mirrored at b border non-existent
                            f += qkl * fB * aB(ix+dmax+1, iy+dmax+1);

                        if(x - 1 - ix >= input.getXMin() && x - 1 - ix <= input.getXMax())
                            // don't apply shift if pixel mirrored at l border non-existent
                            f += qkl * fL * aL(ix+dmax+1, iy+dmax+1);

                        if(x + 1 - ix >= input.getXMin() && x + 1 - ix <= input.getXMax())
                            // don't apply shift if pixel mirrored at r border non-existent
                            f += qkl * fR * aR(ix+dmax+1, iy+dmax+1);

                    }
                }
                output(x, y) = f;
            }
        }
    }

    // instantiate template functions for expected types: float and double currently
    template void ApplyCD(ImageView<double>& output, const BaseImage<double>& input,
                          const BaseImage<double>& aL, const BaseImage<double>& aR,
                          const BaseImage<double>& aB, const BaseImage<double>& aT,
                          const int dmax, const double gain_ratio);
    template void ApplyCD(ImageView<float>& output, const BaseImage<float>& input,
                          const BaseImage<double>& aL, const BaseImage<double>& aR,
                          const BaseImage<double>& aB, const BaseImage<double>& aT,
                          const int dmax, const double gain_ratio);
}
