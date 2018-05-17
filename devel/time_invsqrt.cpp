/* -*- c++ -*-
 * Copyright (c) 2012-2018 by the GalSim developers team on GitHub
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

#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <xmmintrin.h>
#include <sys/time.h>
#include "../include/galsim/fmath/fmath.hpp"


void sse_rsqrt(float* x, float* y)
{
    __m128 in = _mm_load_ss(x);
    _mm_store_ss(y, _mm_rsqrt_ss(in) );
}

// plus one newton raphson step
void sse_rsqrt_nr(float* x, float* y)
{
    const float threehalfs = 1.5F;
    const float halfx = *x * 0.5F;
    __m128 in = _mm_load_ss(x);
    float yy;
    _mm_store_ss(&yy, _mm_rsqrt_ss( in ) );
    *y = yy * (threehalfs - halfx * yy * yy);
}

void carmack_rsqrt(float* x, float* y)
{
    const float threehalfs = 1.5F;
    const float halfx = *x * 0.5F;
    float yy = *x;
    int32_t i  = *(int32_t *)(&yy);
    i  = 0x5f3759df - (i >> 1);
    yy = *(float *)(&i);
    *y = yy * (threehalfs - halfx * yy * yy);
}

// with extra newton raphson step
void carmack_rsqrt_nr2(float* x, float* y)
{
    const float threehalfs = 1.5F;
    const float halfx = *x * 0.5F;
    float yy  = *x;
    int32_t i  = *(int32_t *)(&yy);
    i  = 0x5f3759df - (i >> 1);
    yy = *(float *)(&i);
    yy *= threehalfs - halfx * yy * yy;
    *y = yy * (threehalfs - halfx * yy * yy);
}

void std_invsqrt(float* x, float* y)
{ *y = 1.F/sqrtf(*x); }

void double_invsqrt(float* x, float* y)
{ *y = 1./sqrt((double)*x); }

void sse4_rsqrt(float* x, float* y)
{
    __m128 in = _mm_loadu_ps(x);
    _mm_storeu_ps(y, _mm_rsqrt_ps(in) );
}

void sse4_rsqrt_nr(float* x, float* y)
{
    const __m128 threehalfs = _mm_set_ps1(1.5F);
    const __m128 half = _mm_set_ps1(0.5F);

    __m128 x4 = _mm_loadu_ps(x);
    __m128 halfx = _mm_mul_ps(half,x4);   // 0.5 * x
    __m128 yy = _mm_rsqrt_ps(x4);
    __m128 tmp = _mm_mul_ps(yy,yy);  // yy * yy
    __m128 tmp2 = _mm_mul_ps(halfx,tmp);  // 0.5 * x * y * y
    tmp = _mm_sub_ps(threehalfs,tmp2);
    tmp2 = _mm_mul_ps(yy, tmp);
    _mm_storeu_ps(y, tmp2);
}

void sse4_invsqrt(float* x, float* y)
{
    const __m128 one = _mm_set_ps1(1.0F);

    __m128 x4 = _mm_loadu_ps(x);
    __m128 sqrtx = _mm_sqrt_ps(x4);
    __m128 invsqrtx = _mm_div_ps(one,sqrtx);
    _mm_storeu_ps(y, invsqrtx);
}

double time_loop(long n, float* input, float* output, void (*func)(float*,float*), int inc)
{
    int N = 20;
    double t = 0.;
    for (int i=0; i<N; ++i) {
        // Flush the cache.
        // Taken from http://stackoverflow.com/questions/3446138/how-to-clear-cpu-l1-and-l2-cache
        const int size = 4*1024*1024; // Allocate 4M.  L3 on my system is 3M
        char *c = (char *)malloc(size);
        for (int i = 0; i < 10; i++)
            for (int j = 0; j < size; j++)
                c[j] = (char) (i*c[(int)(c[j]) % size]);

        // Time the loop we care about.
        struct timeval stop, start;
        gettimeofday(&start, NULL);
        for (long i=0; i<n; i+=inc) {
            (*func)(input+i, output+i);
        }
        gettimeofday(&stop, NULL);
        t += (stop.tv_usec - start.tv_usec)/1.e6;
    }
    t /= N;
    return t;
}

double max_err(long n, float* val, double* truth, bool rel_err=false)
{
    double max = 0.;
    long imax = 0;
    for (long i=0; i<n; ++i) {
        double err = fabs(val[i] - truth[i]);
        if (rel_err) err /= fabs(truth[i]);
        if (err > max) { max = err; imax = i; }
    }
    //printf("max at i=%ld: val=%lf, truth=%lf\n",imax,val[imax],truth[imax]);
    return max;
}

void time_sqrt()
{
    // We'll be using this as invsqrt((1+k^2)^3), so arg >= 1
    long n = 100000;
    float input[n], r1[n], r2[n], r3[n], r4[n], r5[n], r6[n], r7[n], r8[n], r9[n];
    double truth[n];

    for (long i=0; i<n; ++i) {
        input[i] = 1.0 + i/100.;
        truth[i] = 1.0 / sqrt(input[i]);
    }

    //printf("sizeof(long): %ld\n", sizeof(long));
    double t1 = time_loop(n, input, r1, &carmack_rsqrt, 1);
    double t2 = time_loop(n, input, r2, &carmack_rsqrt_nr2, 1);
    double t3 = time_loop(n, input, r3, &sse_rsqrt, 1);
    double t4 = time_loop(n, input, r4, &sse_rsqrt_nr, 1);
    double t5 = time_loop(n, input, r5, &std_invsqrt, 1);
    double t6 = time_loop(n, input, r6, &double_invsqrt, 1);
    double t7 = time_loop(n, input, r7, &sse4_rsqrt, 4);
    double t8 = time_loop(n, input, r8, &sse4_rsqrt_nr, 4);
    double t9 = time_loop(n, input, r9, &sse4_invsqrt, 4);

    double err1 = max_err(n, r1, truth);
    double err2 = max_err(n, r2, truth);
    double err3 = max_err(n, r3, truth);
    double err4 = max_err(n, r4, truth);
    double err5 = max_err(n, r5, truth);
    double err6 = max_err(n, r6, truth);
    double err7 = max_err(n, r7, truth);
    double err8 = max_err(n, r8, truth);
    double err9 = max_err(n, r9, truth);

    printf("carmack 1 nr: %g msec, max error = %g\n", t1*1.e3, err1);
    printf("carmack 2 nr: %g msec, max error = %g\n", t2*1.e3, err2);
    printf("sse:          %g msec, max error = %g\n", t3*1.e3, err3);
    printf("sse 1 nr:     %g msec, max error = %g\n", t4*1.e3, err4);
    printf("1./sqrt(x):   %g msec, max error = %g\n", t5*1.e3, err5);
    printf("1./sqrtd(x):  %g msec, max error = %g\n", t6*1.e3, err6);
    printf("sse4 rsqrt:   %g msec, max error = %g\n", t7*1.e3, err7);
    printf("sse4 1 nr:    %g msec, max error = %g\n", t8*1.e3, err8);
    printf("sse4 invsqrt: %g msec, max error = %g\n", t9*1.e3, err9);
}

void std_exp(float* x, float* y)
{
    *y = exp(*x);
}

void fmath_exp(float* x, float* y)
{
    *y = fmath::exp(*x);
}

void fmath_expd(float* x, float* y)
{
    *y = fmath::expd(*x);
}

void fmath_exp_ps(float* x, float* y)
{
    __m128 x4 = _mm_loadu_ps(x);
    __m128 y4 = fmath::exp_ps(x4);
    _mm_storeu_ps(y, y4);
}


void time_exp()
{
    long n = 10000;
    float input[n], e1[n], e2[n], e3[n], e4[n];
    double truth[n];

    for (long i=0; i<n; i+=2) {
        input[i] = -i/200.;
        truth[i] = exp(input[i]);
        input[i+1] = i/200.;
        truth[i+1] = exp(input[i+1]);
    }
    printf("max arg = %lf, exp = %le\n",input[n-1],truth[n-1]);

    double t1 = time_loop(n, input, e1, &std_exp, 1);
    double t2 = time_loop(n, input, e2, &fmath_exp, 1);
    double t3 = time_loop(n, input, e3, &fmath_expd, 1);
    double t4 = time_loop(n, input, e4, &fmath_exp_ps, 4);

    double err1 = max_err(n, e1, truth, true);
    double err2 = max_err(n, e2, truth, true);
    double err3 = max_err(n, e3, truth, true);
    double err4 = max_err(n, e4, truth, true);

    printf("std::exp      %g msec, max error = %g\n", t1*1.e3, err1);
    printf("fmath::exp    %g msec, max error = %g\n", t2*1.e3, err2);
    printf("fmath::expd   %g msec, max error = %g\n", t3*1.e3, err3);
    printf("fmath::exp_ps %g msec, max error = %g\n", t4*1.e3, err4);
}

int main()
{
    //time_sqrt();
    time_exp();
}
