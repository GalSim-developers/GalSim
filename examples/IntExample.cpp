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

//
// Here are some concrete examples of the use of int1d and int2d.
// 
// First, I include the examples from the comment at the beginning of
// the file Int.h.
//
// Next, I include an astronomically useful calculation of coordinate distance
// as a function of redshit.
// 
// If you want more details, see the comment at the beginning of Int.h.
//

#include "GalSim.h"
#include <iostream>
#include <stdexcept>

// A simple Gaussian, parametrized by its center (mu) and size (sigma).
class Gauss : public std::unary_function<double,double>
{
public :

    Gauss(double _mu, double _sig) : 
        mu(_mu), sig(_sig), sigsq(_sig*_sig) {}

    double operator()(double x) const
    { 
        const double SQRTTWOPI = 2.50662827463;
        return exp(-pow(x-mu,2)/2./sigsq)/SQRTTWOPI/sig; 
    }

private :
    double mu,sig,sigsq;
};

// In the file Int.h, I present this as a class Integrand.
// Here I do it as a function to show how that can work just as well.
double foo(double x, double y)
{
    // A simple function:
    // f(x,y) = x*(3*x+y) + y
    return x * (3.*x + y) + y;
}


// This is stripped down from a more complete Cosmology class that
// calculates all kinds of things, including power spectra and such.
// The simplest integration calculation is the w(z) function, so that's 
// all that is replicated here.
struct Cosmology
{
    Cosmology(double _om_m, double _om_v, double _w, double _wa) :
        om_m(_om_m), om_v(_om_v), w(_w), wa(_wa) {}

    double calc_w(double z);
    // calculate coordinate distance (in units of c/Ho) as a function of z.

    double om_m, om_v, w, wa;
};

struct W_Integrator : public std::unary_function<double,double>
{
    W_Integrator(const Cosmology& _c) : c(_c) {}
    double operator()(double a) const
    {
        // First calculate H^2 according to:
        //
        // H^2 = H0^2 * [ Om_m a^-3 + Om_k a^-2 + 
        //                Om_DE exp (-3 [ (1+w+wa)*lna + wa*(1-a) ] ) ]
        // Ignore the H0^2 scaling

        double lna = log(a);
        double hsq = c.om_m * exp(-3.*lna);
        double om_k = 1.-c.om_m-c.om_v;
        if (om_k != 0.) hsq += om_k * exp(-2.*lna);
        if (c.wa == 0.) 
            if (c.w == -1.)
                hsq += c.om_v;
            else
                hsq += c.om_v * exp(-3.*(1.+c.w)*lna);
        else
            hsq += c.om_v * exp(-3.*( (1.+c.w+c.wa)*lna + c.wa*(1.-a) ) );

        if (hsq <= 0.) {
            // This can happen for very strange w, wa values with non-flat 
            // cosmologies so do something semi-graceful if it does.
            std::cerr<<"Invalid hsq for a = "<<a<<".  hsq = "<<hsq<<std::endl;
            throw std::runtime_error("Negative hsq found.");
        }

        // w = int( 1/sqrt(H(z)) dz ) = int( 1/sqrt(H(a)) 1/a^2 da )
        // So we return the integrand.
        return 1./(sqrt(hsq)*(a*a));
    }
    const Cosmology& c;
};

double Cosmology::calc_w(double z)
{
    // w = int( 1/sqrt(H(z)) dz ) = int( 1/sqrt(H(a)) 1/a^2 da )
    // H(a) = H0 sqrt( Om_m a^-3 + Om_k a^-2 +
    //                 Om_de exp(3 int(1+w(a') dln(a'), a'=a..1) ) )
    // For w(a) = w0 + wa(1-a), we can do the internal integral:
    // ... Om_de exp( -3(1+w0+wa) ln(a) - 3 wa(1-a) )

    W_Integrator winteg(*this);

    return galsim::integ::int1d(winteg,1./(1.+z),1);
}

int main()
{
    using galsim::integ::int1d;
    using galsim::integ::int2d;
    using galsim::integ::MOCK_INF;

    // First some integrations of a Gaussian:

    Gauss g01(0.,1.); // mu = 0, sigma = 1.
    Gauss g02(0.,2.); // mu = 0, sigma = 2.

    std::cout<<"int(Gauss(0.,1.) , -1..1) = "<<int1d(g01,-1.,1.)<<std::endl;;
    std::cout<<"int(Gauss(0.,2.) , -1..1) = "<<int1d(g02,-1.,1.)<<std::endl;;

    std::cout<<"int(Gauss(0.,1.) , -2..2) = "<<int1d(g01,-2.,2.)<<std::endl;;
    std::cout<<"int(Gauss(0.,2.) , -2..2) = "<<int1d(g02,-2.,2.)<<std::endl;;

    std::cout<<"int(Gauss(0.,1.) , 0..inf) = "<<int1d(g01,0.,MOCK_INF)<<std::endl;;
    std::cout<<"int(Gauss(0.,2.) , 0..inf) = "<<int1d(g02,0.,MOCK_INF)<<std::endl;;

    std::cout<<"\nint(x*(3*x+y)+y, 0..1, 0..1) = "<<
        int2d(std::ptr_fun(foo),0.,1.,0.,1.)<<std::endl;

    std::cout<<"\nIn a universe with:\n\n";
    std::cout<<"Omega_m = 0.3\n";
    std::cout<<"Omega_v = 0.65\n";
    std::cout<<"w = -0.9\n";
    std::cout<<"wa = 0.2\n";
    std::cout<<"\nThe w(z) relation is:\n\n";
    std::cout<<"z\tw\n\n";
    Cosmology c(0.3,0.65,-0.9,0.2);
    for(double z = 0.; z < 5.01; z += 0.2) {
        std::cout<<z<<"\t"<<c.calc_w(z)<<std::endl;
    }
}
