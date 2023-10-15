/* -*- c++ -*-
 * Copyright (c) 2012-2023 by the GalSim developers team on GitHub
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
#include <numpy/random/bitgen.h>  // For bitgen_t struct

// For reference:
/*
typedef struct bitgen {
  void *state;
  uint64_t (*next_uint64)(void *st);
  uint32_t (*next_uint32)(void *st);
  double (*next_double)(void *st);
  uint64_t (*next_raw)(void *st);
} bitgen_t;
*/

namespace galsim {

    uint32_t next_uint32(void* st)
    {
        BaseDeviate* rng = static_cast<BaseDeviate*>(st);
        return (uint32_t) rng->raw();
    }

    uint64_t next_uint64(void* st)
    {
        uint32_t i1 = next_uint32(st);
        uint32_t i2 = next_uint32(st);
        return ((uint64_t) i1 << 32) + i2;
    }

    double next_double(void* st)
    {
        return next_uint32(st) / ((double) std::numeric_limits<uint32_t>::max()+1.);
    }

    void SetupBitGen(BaseDeviate* rng, py::capsule capsule)
    {
        bitgen_t* bg(capsule);
        bg->state = rng;
        bg->next_uint64 = next_uint64;
        bg->next_uint32 = next_uint32;
        bg->next_double = next_double;
        bg->next_raw = next_uint64;
    };

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

    void pyExportRandom(py::module& _galsim)
    {
        py::class_<BaseDeviate> (_galsim, "BaseDeviateImpl")
            .def(py::init<long>())
            .def(py::init<const BaseDeviate&>())
            .def(py::init<const char*>())
            .def("duplicate", &BaseDeviate::duplicate)
            .def("seed", (void (BaseDeviate::*) (long) )&BaseDeviate::seed)
            .def("reset", (void (BaseDeviate::*) (const BaseDeviate&) )&BaseDeviate::reset)
            .def("clearCache", &BaseDeviate::clearCache)
            .def("serialize", &BaseDeviate::serialize)
            .def("discard", &BaseDeviate::discard)
            .def("raw", &BaseDeviate::raw)
            .def("generate", &Generate)
            .def("add_generate", &AddGenerate)
            .def("setup_bitgen", &SetupBitGen)
            ;

        py::class_<UniformDeviate, BaseDeviate>(
            _galsim, "UniformDeviateImpl")
            .def(py::init<const BaseDeviate&>())
            .def("duplicate", &UniformDeviate::duplicate)
            .def("generate1", &UniformDeviate::generate1);

        py::class_<GaussianDeviate, BaseDeviate>(
            _galsim, "GaussianDeviateImpl")
            .def(py::init<const BaseDeviate&, double, double>())
            .def("duplicate", &GaussianDeviate::duplicate)
            .def("generate1", &GaussianDeviate::generate1)
            .def("generate_from_variance", &GenerateFromVariance);

        py::class_<BinomialDeviate, BaseDeviate>(
            _galsim, "BinomialDeviateImpl")
            .def(py::init<const BaseDeviate&, int, double>())
            .def("duplicate", &BinomialDeviate::duplicate)
            .def("generate1", &BinomialDeviate::generate1);

        py::class_<PoissonDeviate, BaseDeviate>(
            _galsim, "PoissonDeviateImpl")
            .def(py::init<const BaseDeviate&, double>())
            .def("duplicate", &PoissonDeviate::duplicate)
            .def("generate1", &PoissonDeviate::generate1)
            .def("generate_from_expectation", &GenerateFromExpectation);

        py::class_<WeibullDeviate, BaseDeviate>(
            _galsim, "WeibullDeviateImpl")
            .def(py::init<const BaseDeviate&, double, double>())
            .def("duplicate", &WeibullDeviate::duplicate)
            .def("generate1", &WeibullDeviate::generate1);

        py::class_<GammaDeviate, BaseDeviate>(
            _galsim, "GammaDeviateImpl")
            .def(py::init<const BaseDeviate&, double, double>())
            .def("duplicate", &GammaDeviate::duplicate)
            .def("generate1", &GammaDeviate::generate1);

        py::class_<Chi2Deviate, BaseDeviate>(
            _galsim, "Chi2DeviateImpl")
            .def(py::init<const BaseDeviate&, double>())
            .def("duplicate", &Chi2Deviate::duplicate)
            .def("generate1", &Chi2Deviate::generate1);
    }

} // namespace galsim
