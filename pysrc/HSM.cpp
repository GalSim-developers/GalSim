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

#include "PyBind11Helper.h"
#include "hsm/PSFCorr.h"

namespace galsim {
namespace hsm {

    static ShapeData* ShapeData_init(
        const galsim::Bounds<int>& image_bounds, int moments_status,
        float observed_e1, float observed_e2,
        float moments_sigma, float moments_amp,
        const galsim::Position<double>& moments_centroid,
        double moments_rho4, int moments_n_iter,
        int correction_status, float corrected_e1, float corrected_e2,
        float corrected_g1, float corrected_g2, const char* meas_type,
        float corrected_shape_err, const char* correction_method,
        float resolution_factor, float psf_sigma,
        float psf_e1, float psf_e2, const char* error_message)
    {
        ShapeData* data = new ShapeData();
        data->image_bounds = image_bounds;
        data->moments_status = moments_status;
        data->observed_e1 = observed_e1;
        data->observed_e2 = observed_e2;
        data->moments_sigma = moments_sigma;
        data->moments_amp = moments_amp;
        data->moments_centroid = moments_centroid;
        data->moments_rho4 = moments_rho4;
        data->moments_n_iter = moments_n_iter;
        data->correction_status = correction_status;
        data->corrected_e1 = corrected_e1;
        data->corrected_e2 = corrected_e2;
        data->corrected_g1 = corrected_g1;
        data->corrected_g2 = corrected_g2;
        data->meas_type = meas_type;
        data->corrected_shape_err = corrected_shape_err;
        data->correction_method = correction_method;
        data->resolution_factor = resolution_factor;
        data->psf_sigma = psf_sigma;
        data->psf_e1 = psf_e1;
        data->psf_e2 = psf_e2;
        data->error_message = error_message;
        return data;
    }

    template <typename T, typename V>
    static void WrapTemplates(PY_MODULE& _galsim)
    {
        typedef void (*FAM_func)(ShapeData&, const BaseImage<T>&, const BaseImage<int>&,
                                 double, double, Position<double>, bool, const HSMParams&);
        GALSIM_DOT def("_FindAdaptiveMomView", FAM_func(&FindAdaptiveMomView));

        typedef void (*ESH_func)(ShapeData&, const BaseImage<T>&, const BaseImage<V>&,
                                 const BaseImage<int>&, float, const char *,
                                 const char*, double, double, double, Position<double>,
                                 const HSMParams&);
        GALSIM_DOT def("_EstimateShearView", ESH_func(&EstimateShearView));
    };

    void pyExportHSM(PY_MODULE& _galsim)
    {
        py::class_<HSMParams>(GALSIM_COMMA "HSMParams" BP_NOINIT)
            .def(py::init<
                 double, double, double, int, int, double, long, long, double, double, double,
                 int, double, double, double>());

        py::class_<ShapeData>(GALSIM_COMMA "ShapeData" BP_NOINIT)
            .def(PY_INIT(&ShapeData_init))
            .def_readonly("image_bounds", &ShapeData::image_bounds)
            .def_readonly("moments_status", &ShapeData::moments_status)
            .def_readonly("observed_e1", &ShapeData::observed_e1)
            .def_readonly("observed_e2", &ShapeData::observed_e2)
            .def_readonly("moments_sigma", &ShapeData::moments_sigma)
            .def_readonly("moments_amp", &ShapeData::moments_amp)
            .def_readonly("moments_centroid", &ShapeData::moments_centroid)
            .def_readonly("moments_rho4", &ShapeData::moments_rho4)
            .def_readonly("moments_n_iter", &ShapeData::moments_n_iter)
            .def_readonly("correction_status", &ShapeData::correction_status)
            .def_readonly("corrected_e1", &ShapeData::corrected_e1)
            .def_readonly("corrected_e2", &ShapeData::corrected_e2)
            .def_readonly("corrected_g1", &ShapeData::corrected_g1)
            .def_readonly("corrected_g2", &ShapeData::corrected_g2)
            .def_readonly("meas_type", &ShapeData::meas_type)
            .def_readonly("corrected_shape_err", &ShapeData::corrected_shape_err)
            .def_readonly("correction_method", &ShapeData::correction_method)
            .def_readonly("resolution_factor", &ShapeData::resolution_factor)
            .def_readonly("psf_sigma", &ShapeData::psf_sigma)
            .def_readonly("psf_e1", &ShapeData::psf_e1)
            .def_readonly("psf_e2", &ShapeData::psf_e2)
            .def_readonly("error_message", &ShapeData::error_message);

        WrapTemplates<float, float>(_galsim);
        WrapTemplates<double, double>(_galsim);
        WrapTemplates<double, float>(_galsim);
        WrapTemplates<float, double>(_galsim);
    }

} // namespace hsm
} // namespace galsim
