#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <xmmintrin.h>
#include <sys/time.h>


float sse_rsqrt(float x)
{
    float y;
    __m128 in = _mm_load_ss( &x );
    _mm_store_ss( &y, _mm_rsqrt_ss( in ) );
    return y;
}

// plus one newton raphson step
float sse_rsqrt_nr(float x)
{
    const float threehalfs = 1.5F;
    float y, x2;

    x2 = x * 0.5F;

    __m128 in = _mm_load_ss( &x );
    _mm_store_ss( &y, _mm_rsqrt_ss( in ) );

    return y * ( threehalfs - ( x2 * y * y ) );
}


float carmack_rsqrt(float x)
{
    int32_t i;
    float x2, y;
    const float threehalfs = 1.5F;

    x2 = x * 0.5F;
    y  = x;
    i  = * ( int32_t * ) &y;
    i  = 0x5f3759df - ( i >> 1 );
    y  = * ( float * ) &i;
    return y * ( threehalfs - ( x2 * y * y ) );
}

// with extra newton raphson step
float carmack_rsqrt_nr2(float x)
{
    int32_t i;
    float x2, y;
    const float threehalfs = 1.5F;

    x2 = x * 0.5F;
    y  = x;
    i  = * ( int32_t * ) &y;
    i  = 0x5f3759df - ( i >> 1 );
    y  = * ( float * ) &i;
    y  = y * ( threehalfs - ( x2 * y * y ) );
    return y * ( threehalfs - ( x2 * y * y ) );
}

float std_invsqrt(float x)
{ return 1./sqrt(x); }

float double_invsqrt(float x)
{ return 1./sqrt((double)x); }

double time_loop(long n, float* input, float* output, float (*func)(float))
{
    struct timeval stop, start;
    gettimeofday(&start, NULL);
    for (long i=0; i<n; ++i) {
        output[i] = (*func)(input[i]);
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
    long n = 1000000;
    float input[n], truth[n], r1[n], r2[n], r3[n], r4[n], r5[n], r6[n];

    for (long i=0; i<n; ++i) {
        input[i] = 1.0 + i/100.;
        truth[i] = 1.0 / sqrt(input[i]);
    }

    //printf("sizeof(long): %ld\n", sizeof(long));
    double t1 = time_loop(n, input, r1, &carmack_rsqrt);
    double t2 = time_loop(n, input, r2, &carmack_rsqrt_nr2);
    double t3 = time_loop(n, input, r3, &sse_rsqrt);
    double t4 = time_loop(n, input, r4, &sse_rsqrt_nr);
    double t5 = time_loop(n, input, r5, &std_invsqrt);
    double t6 = time_loop(n, input, r6, &double_invsqrt);

    double err1 = max_err(n, r1, truth);
    double err2 = max_err(n, r2, truth);
    double err3 = max_err(n, r3, truth);
    double err4 = max_err(n, r4, truth);
    double err5 = max_err(n, r5, truth);
    double err6 = max_err(n, r6, truth);

    printf("carmack 1 nr: %g msec, max error = %g\n", t1*1.e3, err1);
    printf("carmack 2 nr: %g msec, max error = %g\n", t2*1.e3, err2);
    printf("sse:          %g msec, max error = %g\n", t3*1.e3, err3);
    printf("sse 1 nr:     %g msec, max error = %g\n", t4*1.e3, err4);
    printf("1./sqrt(x):   %g msec, max error = %g\n", t5*1.e3, err5);
    printf("1./sqrtd(x):  %g msec, max error = %g\n", t6*1.e3, err6);

}
