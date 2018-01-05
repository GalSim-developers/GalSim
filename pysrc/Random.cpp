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

#include "PyBind11Helper.h"
#include "Random.h"

namespace galsim {

    void Generate(BaseDeviate& rng, size_t N, size_t idata)
    {
        double* data = reinterpret_cast<double*>(idata);
        rng.generate(N, data);
    }

    void AddGenerate(BaseDeviate& rng, size_t N, size_t idata)
    {
        double* data = reinterpret_cast<double*>(idata);
        rng.addGenerate(N, data);
    }

    void GenerateFromVariance(GaussianDeviate& rng, size_t N, size_t idata)
    {
        double* data = reinterpret_cast<double*>(idata);
        rng.generateFromVariance(N, data);
    }

    void GenerateFromExpectation(PoissonDeviate& rng, size_t N, size_t idata)
    {
        double* data = reinterpret_cast<double*>(idata);
        rng.generateFromExpectation(N, data);
    }

    void pyExportRandom(PB11_MODULE& _galsim)
    {
        bp::class_<BaseDeviate> (GALSIM_COMMA "BaseDeviateImpl" BP_NOINIT)
            .def(bp::init<long>())
            .def(bp::init<const BaseDeviate&>())
            .def(bp::init<const char*>())
            .def("seed", (void (BaseDeviate::*) (long) )&BaseDeviate::seed)
            .def("reset", (void (BaseDeviate::*) (const BaseDeviate&) )&BaseDeviate::reset)
            .def("clearCache", &BaseDeviate::clearCache)
            .def("serialize", &BaseDeviate::serialize)
            .def("discard", &BaseDeviate::discard)
            .def("raw", &BaseDeviate::raw)
            .def("generate", &Generate)
            .def("add_generate", &AddGenerate);

        bp::class_<UniformDeviate BP_BASES(BaseDeviate)>(
            GALSIM_COMMA "UniformDeviateImpl" BP_NOINIT)
            .def(bp::init<const BaseDeviate&>())
            .def("generate1", &UniformDeviate::generate1);

        bp::class_<GaussianDeviate BP_BASES(BaseDeviate)>(
            GALSIM_COMMA "GaussianDeviateImpl" BP_NOINIT)
            .def(bp::init<const BaseDeviate&, double, double>())
            .def("generate1", &GaussianDeviate::generate1)
            .def("generate_from_variance", &GenerateFromVariance);

        bp::class_<BinomialDeviate BP_BASES(BaseDeviate)>(
            GALSIM_COMMA "BinomialDeviateImpl" BP_NOINIT)
            .def(bp::init<const BaseDeviate&, int, double>())
            .def("generate1", &BinomialDeviate::generate1);

        bp::class_<PoissonDeviate BP_BASES(BaseDeviate)>(
            GALSIM_COMMA "PoissonDeviateImpl" BP_NOINIT)
            .def(bp::init<const BaseDeviate&, double>())
            .def("generate1", &PoissonDeviate::generate1)
            .def("generate_from_expectation", &GenerateFromExpectation);

        bp::class_<WeibullDeviate BP_BASES(BaseDeviate)>(
            GALSIM_COMMA "WeibullDeviateImpl" BP_NOINIT)
            .def(bp::init<const BaseDeviate&, double, double>())
            .def("generate1", &WeibullDeviate::generate1);

        bp::class_<GammaDeviate BP_BASES(BaseDeviate)>(
            GALSIM_COMMA "GammaDeviateImpl" BP_NOINIT)
            .def(bp::init<const BaseDeviate&, double, double>())
            .def("generate1", &GammaDeviate::generate1);

        bp::class_<Chi2Deviate BP_BASES(BaseDeviate)>(
            GALSIM_COMMA "Chi2DeviateImpl" BP_NOINIT)
            .def(bp::init<const BaseDeviate&, double>())
            .def("generate1", &Chi2Deviate::generate1);
    }

} // namespace galsim
