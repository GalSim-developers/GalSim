/* -*- c++ -*-
 * Copyright (c) 2012-2014 by the GalSim developers team on GitHub
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

//#define DEBUGLOGGING

#include "CDModel.h"

#ifdef DEBUGLOGGING
#include <fstream>
std::ostream* dbgout = new std::ofstream("debug.out");
int verbose_level = 2;
/*
 * There are three levels of verbosity which can be helpful when debugging, which are written as
 * dbg, xdbg, xxdbg (all defined in Std.h).
 * It's Mike's way to have debug statements in the code that are really easy to turn on and off.
 *
 * If DEBUGLOGGING is #defined, then these write out to *dbgout, according to the value of 
 * verbose_level.
 * dbg requires verbose_level >= 1
 * xdbg requires verbose_level >= 2
 * xxdbg requires verbose_level >= 3
 * If DEBUGLOGGING is not defined, the all three becomes just `if (false) std::cerr`,
 * so the compiler parses the statement fine, but trivially optimizes the code away, so there is no
 * efficiency hit from leaving them in the code.
 */
#endif

namespace galsim {

template <typename T>
ImageAlloc<T> ApplyCD(BaseImage<T> image, ConstImageView<double> aL, ConstImageView<double> aR, 
                      ConstImageView<double> aB, ConstImageView<double> aT, const int dmax,
                      const double gain_ratio) const
{
    // images aL, aR, aB, aT contain shift coefficients for left, right, bottom and top border 
    // dmax is maximum separation considered
    // gain_ratio is gain_image/gain_flat when shift coefficients were measured on flat and 
    // image has different gain

    // Perform sanity check
    if(dmax < 0) throw ImageError("Attempt to apply CD model with invalid extent");
    // Get the array dimension and perform other checks
    const int arraydim = 1 + aL.getXMax() - aL.getXMin();
    if (arraydim != (2 * dmax + 1) * (2 * dmax + 1)) throw ImageError(
        "Dimensions of input image do not match specified dmax");
    if (1 + aR.getXMax() - aR.getXMin() != arraydim ||
        1 + aB.getXMax() - aB.getXMin() != arraydim ||
        1 + aT.getXMax() - aT.getXMin() != arraydim)
        throw ImageError("All input aL, aR, aB, aT Images must be the same dimensions");
    
    ImageAlloc<T> output(image.getXMax()-image.getXMin()+1, image.getYMax()-image.getYMin()+1);  
    // working version of image, which we later return
    
    for(int x=image.getXMin(); x<=image.getXMax(); x++){

        for(int y=image.getYMin(); y<=image.getYMax(); y++){

            double f = image.at(x, y);
            double fT = 0.; if(y < image.getYMax()) fT = (f + image.at(x, y + 1)) / 2.;
            double fB = 0.; if(y > image.getYMin()) fB = (f + image.at(x, y - 1)) / 2.;
            double fR = 0.; if(x < image.getXMax()) fR = (f + image.at(x + 1, y)) / 2.;
            double fL = 0.; if(x > image.getXMin()) fL = (f + image.at(x - 1, y)) / 2.;

            // for each surrounding pixel do
            int matrixindex = 0; // for iterating over the aL, aR, aB, aT images in 1d

            for(int iy=-dmax; iy<=dmax; iy++){

                for(int ix=-dmax; ix<=dmax; ix++){

                    if(x+ix<image.getXMin() || x+ix>image.getXMax() || y+iy<image.getYMin() ||
                        y+iy>image.getYMax()) {
                        // a non-existent pixel is not going to move us
                        matrixindex++;
                        continue;
                    }
                    double qkl = image.at(x + ix, y + iy) * gain_ratio;

                    if(y + 1 - iy >= image.getYMin() && y + 1 - iy <= image.getYMax())  
                      // don't apply shift if pixel mirrored at t border non-existent
                      f += qkl * fT * aT.at(aT.getXMin() + matrixindex, aT.getYMin());

                    if(y - 1 - iy >= image.getYMin() && y - 1 - iy <= image.getYMax())  
                      // don't apply shift if pixel mirrored at b border non-existent
                      f += qkl * fB * aB.at(aB.getXMin() + matrixindex, aB.getYMin());

                    if(x - 1 - ix >= image.getXMin() && x - 1 - ix <= image.getXMax())  
                      // don't apply shift if pixel mirrored at l border non-existent
                      f += qkl * fL * aL.at(aL.getXMin() + matrixindex, aL.getYMin());

                    if(x + 1 - ix >= image.getXMin() && x + 1 - ix <= image.getXMax())  
                      // don't apply shift if pixel mirrored at r border non-existent
                      f += qkl * fR * aR.at(aR.getXMin() + matrixindex, aR.getYMin());

                    matrixindex++;
                }
            }
            output.setValue(x, y, f);
        }
    }
    return output;
}

// instantiate template functions for expected types: float and double currently
template ImageAlloc<float> ApplyCD(
    BaseImage<float> image, ConstImageView<double> aL, ConstImageView<double> aR,
    ConstImageView<double> aB, ConstImageView<double> aT, const int dmax, const double gain_ratio)
    const;
template ImageAlloc<double> ApplyCD(
    BaseImage<double> image, ConstImageView<double> aL, ConstImageView<double> aR,
    ConstImageView<double> aB, ConstImageView<double> aT, const int dmax, const double gain_ratio)
    const;
}
