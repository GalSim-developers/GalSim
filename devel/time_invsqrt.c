#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <xmmintrin.h>
#include <sys/time.h>


void sse_rsqrt(float* x, float* y)
{
    __m128 in = _mm_load_ss(x);
    _mm_store_ss(y, _mm_rsqrt_ss(in) );
}

// plus one newton raphson step
void sse_rsqrt_nr(float* x, float* y)
{
    const float threehalfs = 1.5F;
    const float x2 = *x * 0.5F;
    __m128 in = _mm_load_ss(x);
    _mm_store_ss(y, _mm_rsqrt_ss( in ) );

    *y *= ( threehalfs - ( x2 * *y * *y ) );
}


void carmack_rsqrt(float* x, float* y)
{
    const float threehalfs = 1.5F;
    const float x2 = *x * 0.5F;
    float yy = *x;
    int32_t i  = *(int32_t *)(&yy);
    i  = 0x5f3759df - (i >> 1);
    yy = *(float *)(&i);
    yy *= ( threehalfs - ( x2 * yy * yy ) );
    *y = yy;
}

// with extra newton raphson step
void carmack_rsqrt_nr2(float* x, float* y)
{
    const float threehalfs = 1.5F;
    const float x2 = *x * 0.5F;
    float yy  = *x;
    int32_t i  = *(int32_t *)(&yy);
    i  = 0x5f3759df - (i >> 1);
    yy = *(float *)(&i);
    yy *= threehalfs - ( x2 * yy * yy );
    yy *= threehalfs - ( x2 * yy * yy );
    *y = yy;
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
    __m128 x2 = _mm_mul_ps(half,x4);   // 0.5 * x
    __m128 y4 = _mm_rsqrt_ps(x4);
    __m128 tmp = _mm_mul_ps(y4,y4);  // y * y
    tmp = _mm_mul_ps(x2,tmp);  // 0.5 * x * y * y
    tmp = _mm_sub_ps(threehalfs,tmp);
    y4 = _mm_mul_ps(y4, tmp);
    _mm_storeu_ps(y, y4);
}

void sse4_invsqrt(float* x, float* y)
{
    const __m128 one = _mm_set_ps1(1.0F);

    __m128 x4 = _mm_loadu_ps(x);
    __m128 y4 = _mm_sqrt_ps(x4);
    y4 = _mm_div_ps(one,y4);
    _mm_storeu_ps(y, y4);
}

double time_loop(long n, float* input, float* output, void (*func)(float*,float*), int inc)
{
    struct timeval stop, start;
    gettimeofday(&start, NULL);
    for (long i=0; i<n; i+=inc) {
        (*func)(input+i, output+i);
    }
    gettimeofday(&stop, NULL);
    return (stop.tv_usec - start.tv_usec)/1.e6;
}

double max_err(long n, float* val, float* truth)
{
    double max = 0.;
    for (long i=0; i<n; ++i) {
        double err = fabs(val[i] - truth[i]);
        if (err > max) max = err;
    }
    return max;
}

int main()
{
    // We'll be using this as invsqrt((1+k^2)^3), so arg >= 1
    long n = 100000;
    float input[n], truth[n], r1[n], r2[n], r3[n], r4[n], r5[n], r6[n], r7[n], r8[n], r9[n];

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
    printf("sse4:         %g msec, max error = %g\n", t7*1.e3, err7);
    printf("sse4 1 nr:    %g msec, max error = %g\n", t8*1.e3, err8);
    printf("sse4 invsqrt: %g msec, max error = %g\n", t9*1.e3, err9);

}
