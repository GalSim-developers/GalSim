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

#ifndef GalSim_CDModel_H
#define GalSim_CDModel_H

/**
 * @file CDModel.h @brief Contains the ApplyCD function for applying the Antilogus et al (2014)
 * charge deflection model to correct for brighter-fatter effects.
 */
#include "Image.h"

namespace galsim {

    /**
     *  @brief Tranform an input image using the Antilogus et al (2014) charge deflection model.
     *
     *  The four (2dmax+1)x(2dmax+1) 'aX' matrices are supplied as flattened ConstImageView objects
     *  that are ordered as, e.g.,
     *  aL(dx=-dmax,dy=-dmax), aL(dx=-dmax+1,dy=-dmax), ..., aT(dx=+dmax,dy=+dmax)
     *
     *  gain_ratio is gain_img/gain_flat when 'aX' matrices were derived from flat field images with
     *  a gain differing from that in the supplied image.
     */
    template <typename T>
    PUBLIC_API void ApplyCD(
        ImageView<T>& output, const BaseImage<T>& input,
        const BaseImage<double>& aL, const BaseImage<double>& aR,
        const BaseImage<double>& aB, const BaseImage<double>& aT,
        const int dmax, const double gain_ratio);
}
#endif
