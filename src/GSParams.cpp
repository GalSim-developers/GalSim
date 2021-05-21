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

#include "GSParams.h"

namespace galsim {

    GSParams::GSParams(int _minimum_fft_size,
                       int _maximum_fft_size,
                       double _folding_threshold,
                       double _stepk_minimum_hlr,
                       double _maxk_threshold,
                       double _kvalue_accuracy,
                       double _xvalue_accuracy,
                       double _table_spacing,
                       double _realspace_relerr,
                       double _realspace_abserr,
                       double _integration_relerr,
                       double _integration_abserr,
                       double _shoot_accuracy):
        minimum_fft_size(_minimum_fft_size),
        maximum_fft_size(_maximum_fft_size),
        folding_threshold(_folding_threshold),
        stepk_minimum_hlr(_stepk_minimum_hlr),
        maxk_threshold(_maxk_threshold),
        kvalue_accuracy(_kvalue_accuracy),
        xvalue_accuracy(_xvalue_accuracy),
        table_spacing(_table_spacing),
        realspace_relerr(_realspace_relerr),
        realspace_abserr(_realspace_abserr),
        integration_relerr(_integration_relerr),
        integration_abserr(_integration_abserr),
        shoot_accuracy(_shoot_accuracy)
    {}

    bool GSParams::operator==(const GSParams& rhs) const
    {
        if (this == &rhs) return true;
        else if (minimum_fft_size != rhs.minimum_fft_size) return false;
        else if (maximum_fft_size != rhs.maximum_fft_size) return false;

        else if (folding_threshold != rhs.folding_threshold) return false;
        else if (stepk_minimum_hlr != rhs.stepk_minimum_hlr) return false;
        else if (maxk_threshold != rhs.maxk_threshold) return false;

        else if (kvalue_accuracy != rhs.kvalue_accuracy) return false;
        else if (xvalue_accuracy != rhs.xvalue_accuracy) return false;
        else if (table_spacing != rhs.table_spacing) return false;

        else if (realspace_relerr != rhs.realspace_relerr) return false;
        else if (realspace_abserr != rhs.realspace_abserr) return false;

        else if (integration_relerr != rhs.integration_relerr) return false;
        else if (integration_abserr != rhs.integration_abserr) return false;

        else if (shoot_accuracy != rhs.shoot_accuracy) return false;
        else return true;
    }

    bool GSParams::operator<(const GSParams& rhs) const
    {
        if (this == &rhs) return false;
        else if (minimum_fft_size < rhs.minimum_fft_size) return true;
        else if (minimum_fft_size > rhs.minimum_fft_size) return false;
        else if (maximum_fft_size < rhs.maximum_fft_size) return true;
        else if (maximum_fft_size > rhs.maximum_fft_size) return false;
        else if (folding_threshold < rhs.folding_threshold) return true;
        else if (folding_threshold > rhs.folding_threshold) return false;
        else if (stepk_minimum_hlr < rhs.stepk_minimum_hlr) return true;
        else if (stepk_minimum_hlr > rhs.stepk_minimum_hlr) return false;
        else if (maxk_threshold < rhs.maxk_threshold) return true;
        else if (maxk_threshold > rhs.maxk_threshold) return false;
        else if (kvalue_accuracy < rhs.kvalue_accuracy) return true;
        else if (kvalue_accuracy > rhs.kvalue_accuracy) return false;
        else if (xvalue_accuracy < rhs.xvalue_accuracy) return true;
        else if (xvalue_accuracy > rhs.xvalue_accuracy) return false;
        else if (table_spacing < rhs.table_spacing) return true;
        else if (table_spacing > rhs.table_spacing) return false;
        else if (realspace_relerr < rhs.realspace_relerr) return true;
        else if (realspace_relerr > rhs.realspace_relerr) return false;
        else if (realspace_abserr < rhs.realspace_abserr) return true;
        else if (realspace_abserr > rhs.realspace_abserr) return false;
        else if (integration_relerr < rhs.integration_relerr) return true;
        else if (integration_relerr > rhs.integration_relerr) return false;
        else if (integration_abserr < rhs.integration_abserr) return true;
        else if (integration_abserr > rhs.integration_abserr) return false;
        else if (shoot_accuracy < rhs.shoot_accuracy) return true;
        else if (shoot_accuracy > rhs.shoot_accuracy) return false;
        else return false;
    }

    std::ostream& operator<<(std::ostream& os, const GSParams& gsp)
    {
        os << gsp.minimum_fft_size << "," << gsp.maximum_fft_size << ",  "
            << gsp.folding_threshold << "," << gsp.stepk_minimum_hlr << ","
            << gsp.maxk_threshold << ",  "
            << gsp.kvalue_accuracy << "," << gsp.xvalue_accuracy << ","
            << gsp.table_spacing << ", "
            << gsp.realspace_relerr << "," << gsp.realspace_abserr << ",  "
            << gsp.integration_relerr << "," << gsp.integration_abserr << ",  "
            << gsp.shoot_accuracy;
        return os;
    }

}
