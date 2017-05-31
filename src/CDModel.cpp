/* -*- c++ -*-
 * Copyright (c) 2012-2017 by the GalSim developers team on GitHub
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
    ImageAlloc<T> ApplyCD(const BaseImage<T> &image, ConstImageView<double> aL,
                          ConstImageView<double> aR, ConstImageView<double> aB,
                          ConstImageView<double> aT, const int dmax, const double gain_ratio)
    {
        // images aL, aR, aB, aT contain shift coefficients for left, right, bottom and top border
        // dmax is maximum separation considered
        // gain_ratio is gain_image/gain_flat when shift coefficients were measured on flat and
        // image has different gain

        // Perform sanity check
        if(dmax < 0) throw ImageError("Attempt to apply CD model with invalid extent");

        ImageAlloc<T> output(image.getBounds());
        // working version of image, which we later return

        // compare eqn. 4.5 in Antilogus+2014
        // output is
        //        (1)   input +
        //        (2)   interpolated version of image at pixel borders *
        //        (3)   image convolved with shift coefficients

        for(int x=image.getXMin(); x<=image.getXMax(); x++){
            for(int y=image.getYMin(); y<=image.getYMax(); y++){

                // (1) input image
                double f = image.at(x, y);

                // (2) interpolated version of image at pixel borders
                double fT = 0.; if(y < image.getYMax()) fT = (f + image.at(x, y + 1)) / 2.;
                double fB = 0.; if(y > image.getYMin()) fB = (f + image.at(x, y - 1)) / 2.;
                double fR = 0.; if(x < image.getXMax()) fR = (f + image.at(x + 1, y)) / 2.;
                double fL = 0.; if(x > image.getXMin()) fL = (f + image.at(x - 1, y)) / 2.;

                // (3) convolution of image with shift coefficient matrix
                for(int iy=-dmax; iy<=dmax; iy++){
                    for(int ix=-dmax; ix<=dmax; ix++){

                        if(x+ix<image.getXMin() || x+ix>image.getXMax() ||
                           y+iy<image.getYMin() || y+iy>image.getYMax()) {
                            continue; // a non-existent pixel is not going to move us
                        }
                        double qkl = image.at(x + ix, y + iy) * gain_ratio;

                        if(y + 1 - iy >= image.getYMin() && y + 1 - iy <= image.getYMax())
                            // don't apply shift if pixel mirrored at t border non-existent
                            f += qkl * fT * aT.at(ix+dmax+1, iy+dmax+1);

                        if(y - 1 - iy >= image.getYMin() && y - 1 - iy <= image.getYMax())
                            // don't apply shift if pixel mirrored at b border non-existent
                            f += qkl * fB * aB.at(ix+dmax+1, iy+dmax+1);

                        if(x - 1 - ix >= image.getXMin() && x - 1 - ix <= image.getXMax())
                            // don't apply shift if pixel mirrored at l border non-existent
                            f += qkl * fL * aL.at(ix+dmax+1, iy+dmax+1);

                        if(x + 1 - ix >= image.getXMin() && x + 1 - ix <= image.getXMax())
                            // don't apply shift if pixel mirrored at r border non-existent
                            f += qkl * fR * aR.at(ix+dmax+1, iy+dmax+1);

                    }
                }
                output.setValue(x, y, f);

            }
        }
        return output;
    }

    // instantiate template functions for expected types: float and double currently
    template ImageAlloc<float> ApplyCD(
        const BaseImage<float> &image, ConstImageView<double> aL, ConstImageView<double> aR,
        ConstImageView<double> aB, ConstImageView<double> aT, const int dmax,
        const double gain_ratio);
    template ImageAlloc<double> ApplyCD(
        const BaseImage<double> &image, ConstImageView<double> aL, ConstImageView<double> aR,
        ConstImageView<double> aB, ConstImageView<double> aT, const int dmax,
        const double gain_ratio);
}
