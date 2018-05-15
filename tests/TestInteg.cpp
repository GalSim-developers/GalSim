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

#include "galsim/integ/Int.h"
#include "Test.h"

const double test_sigma = 7.;           // test value of Gaussian sigma for integral tests
const double test_rel_err = 1.e-7;      // the relative accuracy at which to test
const double test_abs_err = 1.e-13;     // the absolute accuracy at which to test
const double test_mock_inf = 2.e10;     // number large enough to get interpreted as infinity by
                                        // integration routines
// Note: all "true" answers in the below tests are found using Wolfram Alpha.

// A simple Gaussian that works as a functional object
class Gauss : public std::unary_function<double,double>
{
public :
    Gauss(double sig) : _sig(sig) {}

    double operator()(double x) const
    { return exp(-0.5*pow(x/_sig,2)); }

private :
    double _sig;
};

// A simple power law
class Power : public std::unary_function<double,double>
{
public :
    Power(double expon) : _expon(expon) {}

    double operator()(double x) const
    { return pow(x,_expon); }

private :
    double _expon;
};

// A straight function, rather than a functional class:
double osc_func(double x)
{ return sin(pow(x,2)) * exp(-std::abs(x)); }

// A simple function:
// f(x,y) = x*(3*x+y) + y
double twod_func(double x, double y)
{ return x * (3.*x + y) + y; }

void TestGaussian()
{
    Log("Start TestGaussian()");
    Gauss gauss(test_sigma);

    double test1 = galsim::integ::int1d(gauss, -1., 1., test_rel_err, test_abs_err);
    AssertClose(test1, 1.99321805307377285009, test_rel_err, test_abs_err);

    double test2 = galsim::integ::int1d(gauss, 0., 20., test_rel_err, test_abs_err);
    AssertClose(test2, 8.73569586966967345835, test_rel_err, test_abs_err);

    double test3 = galsim::integ::int1d(gauss, -50., -40., test_rel_err, test_abs_err);
    AssertClose(test3, 9.66426031085587421984e-8, test_rel_err, test_abs_err);

    double test4 = galsim::integ::int1d(gauss, 0., test_mock_inf, test_rel_err, test_abs_err);
    AssertClose(test4, 8.77319896120850210849, test_rel_err, test_abs_err);

    double test5 = galsim::integ::int1d(gauss, -test_mock_inf, 5.4, test_rel_err, test_abs_err);
    AssertClose(test5, 13.68221660030048620971, test_rel_err, test_abs_err);

    double test6 = galsim::integ::int1d(gauss, -test_mock_inf, test_mock_inf,
                                        test_rel_err, test_abs_err);
    AssertClose(test6, 17.54639792241700421699, test_rel_err, test_abs_err);
}

void TestOscillatory()
{
    Log("Start TestOscillatory()");
    double test1 = galsim::integ::int1d(std::ptr_fun(osc_func), -1., 1.,
                                        test_rel_err, test_abs_err);
    AssertClose(test1, 0.30182513444548879567, test_rel_err, test_abs_err);

    double test2 = galsim::integ::int1d(std::ptr_fun(osc_func), 0., 20.,
                                        test_rel_err, test_abs_err);
    AssertClose(test2, 0.27051358019041255485, test_rel_err, test_abs_err);

    double test3 = galsim::integ::int1d(std::ptr_fun(osc_func), -15., -14.,
                                        test_rel_err, test_abs_err);
    AssertClose(test3, 7.81648378350593176887e-9, test_rel_err, test_abs_err);

    double test4 = galsim::integ::int1d(std::ptr_fun(osc_func), 0., test_mock_inf,
                                        test_rel_err, test_abs_err);
    AssertClose(test4, 0.27051358016221414426, test_rel_err, test_abs_err);

    double test5 = galsim::integ::int1d(std::ptr_fun(osc_func), -test_mock_inf, 5.4,
                                        test_rel_err, test_abs_err);
    AssertClose(test5, 0.5413229824941895221, test_rel_err, test_abs_err);

    double test6 = galsim::integ::int1d(std::ptr_fun(osc_func), -test_mock_inf, test_mock_inf,
                                        test_rel_err, test_abs_err);
    AssertClose(test6, 0.54102716032442828852, test_rel_err, test_abs_err);
}

void TestPole()
{
    Log("Start TestPole()");
    Power powm05(-0.5);

    double test1 = galsim::integ::int1d(powm05, 0., 1., test_rel_err, test_abs_err);
    AssertClose(test1, 2., test_rel_err, test_abs_err);

    double test2 = galsim::integ::int1d(powm05, 0., 300., test_rel_err, test_abs_err);
    AssertClose(test2, 34.64101615137754587055, test_rel_err, test_abs_err);

    Power powm2(-2.);

    double test3 = galsim::integ::int1d(powm2, 1., 2., test_rel_err, test_abs_err);
    AssertClose(test3, 0.5, test_rel_err, test_abs_err);

    double test4 = galsim::integ::int1d(powm2, 1., test_mock_inf, test_rel_err, test_abs_err);
    AssertClose(test4, 1., test_rel_err, test_abs_err);

#if 0
    // This works if everything uses the same compiler.  But boost testing may have
    // been installed with a different compiler, in which case it is unable to catch
    // exceptions thrown from GalSim.  So we skip this test normally.
    // (Developers working on the integrator should reenable this during development.)
    BOOST_CHECK_THROW(
        galsim::integ::int1d(powm2, 0., 1., test_rel_err, test_abs_err),
        galsim::integ::IntFailure);
#endif
}

void Test2d()
{
    Log("Start Test2d()");
    double test1 = galsim::integ::int2d(std::ptr_fun(twod_func),0.,1.,0.,1.,
                                        test_rel_err, test_abs_err);
    AssertClose(test1, 1.75, test_rel_err, test_abs_err);
}

void TestInteg()
{
    Log("Start tests of galsim::integ");
    TestGaussian();
    TestOscillatory();
    TestPole();
    Test2d();
}
