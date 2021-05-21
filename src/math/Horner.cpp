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

#include "fmath/fmath.hpp"  // For SSE

#include "math/Horner.h"
#include "Std.h"

namespace galsim {
namespace math {

    void HornerStep(const double* x, int n, const double c, double* r)
    {
#ifdef __SSE2__
        for (; n && (!IsAligned(r) || !IsAligned(x)); --n, ++r, ++x)
            *r = *r * *x + c;

        int n2 = n>>1;
        int na = n2<<1;
        n -= na;

        if (n2) {
            __m128d cx = _mm_set1_pd(c);
            __m128d* rx = reinterpret_cast<__m128d*>(r);
            const __m128d* xx = reinterpret_cast<const __m128d*>(x);
            do {
                *rx = _mm_add_pd(_mm_mul_pd(*rx, *xx), cx);
                ++rx;
                ++xx;
            } while (--n2);
        }

        if (n) {
            r += na;
            x += na;
            *r = *r * *x + c;
        }
#else
        for (; n; --n, ++r, ++x) *r = *r * *x + c;
#endif
    }

    void HornerBlock(const double* x, int nx, const double* coef, const double* c, double* result)
    {
        // Repeatedly multiply by x and add next coefficient
        double* r = result;
        for (int i=nx; i; --i) *r++ = *c;
        while(--c >= coef) {
            HornerStep(x, nx, *c, result); // result = result * x + c
        }
        // In the last step, we will have added the constant term, and we're done.
    }

    void Horner(const double* x, int nx, const double* coef, const int nc, double* result)
    {
        // Start at highest power
        const double* c = coef + nc-1;
        // Ignore any trailing zeros
        while (*c == 0. && c > coef) --c;

        // Better for caching to do this in blocks of 64 rather than all at once.
        const int BLOCK_SIZE = 64;
        for(; nx >= BLOCK_SIZE; nx-=BLOCK_SIZE, x+=BLOCK_SIZE, result+=BLOCK_SIZE) {
            HornerBlock(x, BLOCK_SIZE, coef, c, result);
        }
        HornerBlock(x, nx, coef, c, result);
    }

    void HornerStep2(const double* x, int n, const double* t, double* r)
    {
#ifdef __SSE2__
        for (; n && (!IsAligned(r) || !IsAligned(x) || !IsAligned(t)); --n, ++r, ++x, ++t)
            *r = *r * *x + *t;

        int n2 = n>>1;
        int na = n2<<1;
        n -= na;

        if (n2) {
            __m128d* rx = reinterpret_cast<__m128d*>(r);
            const __m128d* xx = reinterpret_cast<const __m128d*>(x);
            const __m128d* tx = reinterpret_cast<const __m128d*>(t);
            do {
                *rx = _mm_add_pd(_mm_mul_pd(*rx, *xx), *tx);
                ++rx;
                ++xx;
                ++tx;
            } while (--n2);
        }

        if (n) {
            r += na;
            x += na;
            t += na;
            *r = *r * *x + *t;
        }
#else
        for (; n; --n, ++r, ++x, ++t) *r = *r * *x + *t;
#endif
    }

    void HornerBlock2(const double* x, const double* y, int nx, const double* coef,
                      const double* c, const int ncy, double* result, double* temp)
    {
        Horner(y, nx, c, ncy, result);
        while((c -= ncy) >= coef) {
            Horner(y, nx, c, ncy, temp);
            HornerStep2(x, nx, temp, result);
        }
    }

    void Horner2D(const double* x, const double* y, int nx,
                  const double* coef, const int ncx, const int ncy,
                  double* result, double* temp)
    {
        const double* c = coef + (ncx-1) * ncy;

        // Better for caching to do this in blocks of 64 rather than all at once.
        const int BLOCK_SIZE = 64;
        for(; nx >= BLOCK_SIZE; nx-=BLOCK_SIZE, x+=BLOCK_SIZE, y+=BLOCK_SIZE, result+=BLOCK_SIZE) {
            HornerBlock2(x, y, BLOCK_SIZE, coef, c, ncy, result, temp);
        }
        HornerBlock2(x, y, nx, coef, c, ncy, result, temp);
    }

}}
