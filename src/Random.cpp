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

#include <sys/time.h>
#include <fcntl.h>
#include <string>
#include <vector>
#include <sstream>
#include <unistd.h>
#include <cstring>  // For memcpy
#include "Random.h"

#include "galsim/IgnoreWarnings.h"

// Variable defined to use a private copy of Boost.Random, modified
// to avoid any reference to Boost.Random elements that might be on
// the local machine.
// Undefine this to use Boost.Random from the local distribution.
#define DIVERT_BOOST_RANDOM

#ifdef DIVERT_BOOST_RANDOM
#include "galsim/boost1_48_0/random/mersenne_twister.hpp"
#include "galsim/boost1_48_0/random/normal_distribution.hpp"
#include "galsim/boost1_48_0/random/binomial_distribution.hpp"
#include "galsim/boost1_48_0/random/poisson_distribution.hpp"
#include "galsim/boost1_48_0/random/uniform_real_distribution.hpp"
#include "galsim/boost1_48_0/random/weibull_distribution.hpp"
#include "galsim/boost1_48_0/random/gamma_distribution.hpp"
#include "galsim/boost1_48_0/random/chi_squared_distribution.hpp"
#else
#include "boost/random/mersenne_twister.hpp"
#include "boost/random/normal_distribution.hpp"
#include "boost/random/binomial_distribution.hpp"
#include "boost/random/poisson_distribution.hpp"
#include "boost/random/uniform_real_distribution.hpp"
#include "boost/random/weibull_distribution.hpp"
#include "boost/random/gamma_distribution.hpp"
#include "boost/random/chi_squared_distribution.hpp"
#endif

namespace galsim {

    struct BaseDeviate::BaseDeviateImpl
    {
        // Note that this class could be templated with the type of Boost.Random generator that
        // you want to use instead of mt19937
        typedef boost::mt19937 rng_type;

        BaseDeviateImpl() : _rng(new rng_type) {}
        shared_ptr<rng_type> _rng;
    };

    BaseDeviate::BaseDeviate() :
        _impl(new BaseDeviateImpl())
    {}

    BaseDeviate::BaseDeviate(long lseed) :
        _impl(new BaseDeviateImpl())
    { seed(lseed); }

    BaseDeviate::BaseDeviate(const BaseDeviate& rhs) :
        _impl(rhs._impl)
    {}

    BaseDeviate::BaseDeviate(const char* str_c) :
        _impl(new BaseDeviateImpl())
    {
        if (str_c == NULL) {
            seed(0);
        } else {
            std::string str(str_c);
            std::istringstream iss(str);
            iss >> *_impl->_rng;
        }
    }

    std::string BaseDeviate::serialize()
    {
        // When serializing, we need to make sure there is no cache being stored
        // by the derived class.
        clearCache();
        std::ostringstream oss;
        oss << *_impl->_rng;
        return oss.str();
    }

    BaseDeviate BaseDeviate::duplicate()
    {
#if 0
        // This is the bespoke, but slow, way to do this.
        return BaseDeviate(serialize().c_str());
#else
        // This is a hack, but it seems to work.  And it's around 100x faster. (!)
        // cf. https://stackoverflow.com/a/16310375/1332281
        // Although in this context, a direct copy is simpler than their suggestion.
        BaseDeviate ret;
        std::memcpy(ret._impl->_rng.get(), _impl->_rng.get(), sizeof(*_impl->_rng));
        return ret;
#endif
    }

    void BaseDeviate::seedurandom()
    {
        // This implementation shamelessly taken from:
        // http://stackoverflow.com/questions/2572366/how-to-use-dev-random-or-urandom-in-c
        int randomData = open("/dev/urandom", O_RDONLY);
        int myRandomInteger;
        size_t randomDataLen = 0;
        while (randomDataLen < sizeof myRandomInteger)
        {
            ssize_t result = read(randomData, ((char*)&myRandomInteger) + randomDataLen,
                                  (sizeof myRandomInteger) - randomDataLen);
            if (result < 0)
                throw std::runtime_error("Unable to read from /dev/urandom");
            randomDataLen += result;
        }
        close(randomData);
        _impl->_rng->seed(myRandomInteger);
    }

    void BaseDeviate::seedtime()
    {
        struct timeval tp;
        gettimeofday(&tp,NULL);
        _impl->_rng->seed(tp.tv_usec);
    }

    void BaseDeviate::seed(long lseed)
    {
        if (lseed == 0) {
            try {
                seedurandom();
            } catch(...) {
                // If urandom is not possible, revert to using the time
                seedtime();
            }
        } else {
            // We often use sequential seeds for our RNG's (so we can be sure that runs on multiple
            // processors are deterministic).  The Boost Mersenne Twister is supposed to work with
            // this kind of seeding, having been updated in April 2005 to address an issue with
            // precisely this sort of re-seeding.
            // (See http://www.boost.org/doc/libs/1_51_0/boost/random/mersenne_twister.hpp).
            // The issue itself is described briefly here:
            // http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/MT2002/emt19937ar.html,
            // and in more detail for an algorithm tt880 that is apparently a 'little cousin' to
            // the Mersenne Twister: http://random.mat.sbg.ac.at/news/seedingTT800.html
            //
            // The worry is that updates to the methods claim improvements to the behaviour of
            // close (in a bitwise sense) patterns, but we have not found ready quantified data.
            //
            // So just to be sure, we send the initial seed through a _different_ random number
            // generator for 2 iterations before using it to seed the RNG we will actually use.
            // This may not be necessary, but it's not much of a performance hit (only occurring on
            // the initial seed of each rng), it can't hurt, and it makes Barney and Mike somewhat
            // less disquieted.  :)

            boost::random::mt11213b alt_rng(lseed);
            alt_rng.discard(2);
            _impl->_rng->seed(alt_rng());
        }
        clearCache();
    }

    void BaseDeviate::reset(long lseed)
    { _impl.reset(new BaseDeviateImpl()); seed(lseed); }

    void BaseDeviate::reset(const BaseDeviate& dev)
    { _impl = dev._impl; clearCache(); }

    void BaseDeviate::discard(int n)
    { _impl->_rng->discard(n); }

    long BaseDeviate::raw()
    { return (*_impl->_rng)(); }

    void BaseDeviate::generate(int N, double* data)
    {
        for (int i=0; i<N; ++i) data[i] = (*this)();
    }

    void BaseDeviate::addGenerate(int N, double* data)
    {
        for (int i=0; i<N; ++i) data[i] += (*this)();
    }

    // Next two functions shamelessly stolen from
    // http://stackoverflow.com/questions/236129/split-a-string-in-c
    std::vector<std::string>& split(const std::string& s, char delim,
                                    std::vector<std::string>& elems)
    {
        std::stringstream ss(s);
        std::string item;
        while (std::getline(ss, item, delim)) {
            elems.push_back(item);
        }
        return elems;
    }

    std::vector<std::string> split(const std::string& s, char delim)
    {
        std::vector<std::string> elems;
        split(s, delim, elems);
        return elems;
    }

    std::string seedstring(const std::vector<std::string>& seed)
    {
        std::ostringstream oss;
        int nseed = seed.size();
        oss << "seed='";
        for (int i=0; i < 3; i++) oss << seed[i] << ' ';
        oss << "...";
        for (int i=nseed-3; i < nseed; i++) oss << ' ' << seed[i];
        oss << "'";
        return oss.str();
    }

    std::string BaseDeviate::make_repr(bool incl_seed)
    {
        // Remember: Don't start with nothing!  See discussion in FormatAndThrow in Std.h
        std::ostringstream oss(" ");
        oss << "galsim.BaseDeviate(";
        if (incl_seed) oss << seedstring(split(serialize(), ' '));
        oss<<")";

        return oss.str();
    }

    struct UniformDeviate::UniformDeviateImpl
    {
        UniformDeviateImpl() : _urd(0., 1.) {}
        boost::random::uniform_real_distribution<> _urd;
    };

    UniformDeviate::UniformDeviate(long lseed) :
        BaseDeviate(lseed), _devimpl(new UniformDeviateImpl()) {}

    UniformDeviate::UniformDeviate(const BaseDeviate& rhs) :
        BaseDeviate(rhs), _devimpl(new UniformDeviateImpl()) {}

    UniformDeviate::UniformDeviate(const UniformDeviate& rhs) :
        BaseDeviate(rhs), _devimpl(rhs._devimpl) {}

    UniformDeviate::UniformDeviate(const char* str_c) :
        BaseDeviate(str_c), _devimpl(new UniformDeviateImpl()) {}

    void UniformDeviate::clearCache() { _devimpl->_urd.reset(); }

    double UniformDeviate::generate1()
    { return _devimpl->_urd(*this->_impl->_rng); }

    std::string UniformDeviate::make_repr(bool incl_seed)
    {
        std::ostringstream oss(" ");
        oss << "galsim.UniformDeviate(";
        if (incl_seed) oss << seedstring(split(serialize(), ' '));
        oss<<")";
        return oss.str();
    }

    struct GaussianDeviate::GaussianDeviateImpl
    {
        GaussianDeviateImpl(double mean, double sigma) : _normal(mean,sigma) {}
        boost::random::normal_distribution<> _normal;
    };

    GaussianDeviate::GaussianDeviate(long lseed, double mean, double sigma) :
        BaseDeviate(lseed), _devimpl(new GaussianDeviateImpl(mean, sigma)) {}

    GaussianDeviate::GaussianDeviate(const BaseDeviate& rhs, double mean, double sigma) :
        BaseDeviate(rhs), _devimpl(new GaussianDeviateImpl(mean, sigma)) {}

    GaussianDeviate::GaussianDeviate(const GaussianDeviate& rhs) :
        BaseDeviate(rhs), _devimpl(rhs._devimpl) {}

    GaussianDeviate::GaussianDeviate(const char* str_c, double mean, double sigma) :
        BaseDeviate(str_c), _devimpl(new GaussianDeviateImpl(mean, sigma)) {}

    double GaussianDeviate::getMean() { return _devimpl->_normal.mean(); }

    double GaussianDeviate::getSigma() { return _devimpl->_normal.sigma(); }

    void GaussianDeviate::setMean(double mean)
    {
        _devimpl->_normal.param(boost::random::normal_distribution<>::param_type(mean,getSigma()));
        clearCache();
    }

    void GaussianDeviate::setSigma(double sigma)
    {
        _devimpl->_normal.param(boost::random::normal_distribution<>::param_type(getMean(),sigma));
        clearCache();
    }

    void GaussianDeviate::clearCache() { _devimpl->_normal.reset(); }

    double GaussianDeviate::generate1()
    { return _devimpl->_normal(*this->_impl->_rng); }

    std::string GaussianDeviate::make_repr(bool incl_seed)
    {
        std::ostringstream oss(" ");
        oss << "galsim.GaussianDeviate(";
        if (incl_seed) oss << seedstring(split(serialize(), ' ')) << ", ";
        oss << "mean="<<getMean()<<", ";
        oss << "sigma="<<getSigma()<<")";
        return oss.str();
    }

    void GaussianDeviate::generateFromVariance(int N, double* data)
    {
        setMean(0.);
        setSigma(1.);
        for (int i=0; i<N; ++i) {
            double sigma = std::sqrt(data[i]);
            data[i] = (*this)() * sigma;
        }
    }

    struct BinomialDeviate::BinomialDeviateImpl
    {
        BinomialDeviateImpl(int N, double p) : _bd(N,p) {}
        boost::random::binomial_distribution<> _bd;
    };

    BinomialDeviate::BinomialDeviate(long lseed, int N, double p) :
        BaseDeviate(lseed), _devimpl(new BinomialDeviateImpl(N,p)) {}

    BinomialDeviate::BinomialDeviate(const BaseDeviate& rhs, int N, double p) :
        BaseDeviate(rhs), _devimpl(new BinomialDeviateImpl(N,p)) {}

    BinomialDeviate::BinomialDeviate(const BinomialDeviate& rhs) :
        BaseDeviate(rhs), _devimpl(rhs._devimpl) {}

    BinomialDeviate::BinomialDeviate(const char* str_c, int N, double p) :
        BaseDeviate(str_c), _devimpl(new BinomialDeviateImpl(N,p)) {}

    int BinomialDeviate::getN() { return _devimpl->_bd.t(); }

    double BinomialDeviate::getP() { return _devimpl->_bd.p(); }

    void BinomialDeviate::setN(int N)
    {
        _devimpl->_bd.param(boost::random::binomial_distribution<>::param_type(N,getP()));
    }

    void BinomialDeviate::setP(double p)
    {
        _devimpl->_bd.param(boost::random::binomial_distribution<>::param_type(getN(),p));
    }

    void BinomialDeviate::clearCache() { _devimpl->_bd.reset(); }

    double BinomialDeviate::generate1()
    { return _devimpl->_bd(*this->_impl->_rng); }

    std::string BinomialDeviate::make_repr(bool incl_seed)
    {
        std::ostringstream oss(" ");
        oss << "galsim.BinomialDeviate(";
        if (incl_seed) oss << seedstring(split(serialize(), ' ')) << ", ";
        oss << "N="<<getN()<<", ";
        oss << "p="<<getP()<<")";
        return oss.str();
    }

    struct PoissonDeviate::PoissonDeviateImpl
    {
        PoissonDeviateImpl(double mean) : _mean(-1) { setMean(mean); }

        double getMean() { return _mean; }

        void setMean(double mean)
        {
            // Near 2**31, the boost poisson rng can wrap around to negative integers, which
            // is bad.  But this high, the Gaussian approximation is extremely accurate, so
            // just use that.
            const double MAX_POISSON = 1<<30;

            if (mean != _mean) {
                _mean = mean;
                if (mean > MAX_POISSON || mean == 0.) setMeanGD(mean);
                else setMeanPD(mean);
            }
        }

        void setMeanGD(double mean)
        {
            _pd.reset();
            if (!_gd) {
                _gd.reset(new boost::random::normal_distribution<>(mean, std::sqrt(mean)));
            } else {
                _gd->param(boost::random::normal_distribution<>::param_type(mean, std::sqrt(mean)));
            }
            _getValue = &PoissonDeviateImpl::getGDValue;
        }

        void setMeanPD(double mean)
        {
            _gd.reset();
            if (!_pd) {
                _pd.reset(new boost::random::poisson_distribution<>(mean));
            } else {
                _pd->param(boost::random::poisson_distribution<>::param_type(mean));
            }
            _getValue = &PoissonDeviateImpl::getPDValue;
        }

        void clearCache()
        {
            if (_pd) _pd->reset();
            if (_gd) _gd->reset();
        }

        typedef BaseDeviate::BaseDeviateImpl::rng_type rng_type;
        double getPDValue(rng_type& rng) { return (*_pd)(rng); }
        double getGDValue(rng_type& rng) { return (*_gd)(rng); }

        double getValue(rng_type& rng)
        { return (this->*_getValue)(rng); }

    private:

        // A variable equal to either getPDValue (normal)
        // or getGDValue (if mean > 2^30)
        double (PoissonDeviateImpl::*_getValue)(rng_type& rng);

        double _mean;
        shared_ptr<boost::random::poisson_distribution<> > _pd;
        shared_ptr<boost::random::normal_distribution<> > _gd;
    };

    PoissonDeviate::PoissonDeviate(long lseed, double mean) :
        BaseDeviate(lseed), _devimpl(new PoissonDeviateImpl(mean)) {}

    PoissonDeviate::PoissonDeviate(const BaseDeviate& rhs, double mean) :
        BaseDeviate(rhs), _devimpl(new PoissonDeviateImpl(mean)) {}

    PoissonDeviate::PoissonDeviate(const PoissonDeviate& rhs) :
        BaseDeviate(rhs), _devimpl(rhs._devimpl) {}

    PoissonDeviate::PoissonDeviate(const char* str_c, double mean) :
        BaseDeviate(str_c), _devimpl(new PoissonDeviateImpl(mean)) {}

    double PoissonDeviate::getMean() { return _devimpl->getMean(); }

    void PoissonDeviate::setMean(double mean) { _devimpl->setMean(mean); }

    double PoissonDeviate::generate1() { return _devimpl->getValue(*this->_impl->_rng); }

    void PoissonDeviate::clearCache() { _devimpl->clearCache(); }

    std::string PoissonDeviate::make_repr(bool incl_seed)
    {
        std::ostringstream oss(" ");
        oss << "galsim.PoissonDeviate(";
        if (incl_seed) oss << seedstring(split(serialize(), ' ')) << ", ";
        oss << "mean="<<getMean()<<")";
        return oss.str();
    }

    void PoissonDeviate::generateFromExpectation(int N, double* data)
    {
        for (int i=0; i<N; ++i) {
            double mean = data[i];
            if (mean > 0.) {
                setMean(mean);
                data[i] = (*this)();
            }
        }
    }

    struct WeibullDeviate::WeibullDeviateImpl
    {
        WeibullDeviateImpl(double a, double b) : _weibull(a,b) {}
        boost::random::weibull_distribution<> _weibull;
    };

    WeibullDeviate::WeibullDeviate(long lseed, double a, double b) :
        BaseDeviate(lseed), _devimpl(new WeibullDeviateImpl(a,b)) {}

    WeibullDeviate::WeibullDeviate(const BaseDeviate& rhs, double a, double b) :
        BaseDeviate(rhs), _devimpl(new WeibullDeviateImpl(a,b)) {}

    WeibullDeviate::WeibullDeviate(const WeibullDeviate& rhs) :
        BaseDeviate(rhs), _devimpl(rhs._devimpl) {}

    WeibullDeviate::WeibullDeviate(const char* str_c, double a, double b) :
        BaseDeviate(str_c), _devimpl(new WeibullDeviateImpl(a,b)) {}

    double WeibullDeviate::getA() { return _devimpl->_weibull.a(); }

    double WeibullDeviate::getB() { return _devimpl->_weibull.b(); }

    void WeibullDeviate::setA(double a)
    {
        _devimpl->_weibull.param(boost::random::weibull_distribution<>::param_type(a,getB()));
    }

    void WeibullDeviate::setB(double b)
    {
        _devimpl->_weibull.param(boost::random::weibull_distribution<>::param_type(getA(),b));
    }

    void WeibullDeviate::clearCache() { _devimpl->_weibull.reset(); }

    double WeibullDeviate::generate1()
    { return _devimpl->_weibull(*this->_impl->_rng); }

    std::string WeibullDeviate::make_repr(bool incl_seed)
    {
        std::ostringstream oss(" ");
        oss << "galsim.WeibullDeviate(";
        if (incl_seed) oss << seedstring(split(serialize(), ' ')) << ", ";
        oss << "a="<<getA()<<", ";
        oss << "b="<<getB()<<")";
        return oss.str();
    }

    struct GammaDeviate::GammaDeviateImpl
    {
        GammaDeviateImpl(double k, double theta) : _gamma(k,theta) {}
        boost::random::gamma_distribution<> _gamma;
    };

    GammaDeviate::GammaDeviate(long lseed, double k, double theta) :
        BaseDeviate(lseed), _devimpl(new GammaDeviateImpl(k,theta)) {}

    GammaDeviate::GammaDeviate(const BaseDeviate& rhs, double k, double theta) :
        BaseDeviate(rhs), _devimpl(new GammaDeviateImpl(k,theta)) {}

    GammaDeviate::GammaDeviate(const GammaDeviate& rhs) :
        BaseDeviate(rhs), _devimpl(rhs._devimpl) {}

    GammaDeviate::GammaDeviate(const char* str_c, double k, double theta) :
        BaseDeviate(str_c), _devimpl(new GammaDeviateImpl(k,theta)) {}

    double GammaDeviate::getK() { return _devimpl->_gamma.alpha(); }

    double GammaDeviate::getTheta() { return _devimpl->_gamma.beta(); }

    void GammaDeviate::setK(double k)
    {
         _devimpl->_gamma.param(boost::random::gamma_distribution<>::param_type(k, getTheta()));
    }

    void GammaDeviate::setTheta(double theta)
    {
         _devimpl->_gamma.param(boost::random::gamma_distribution<>::param_type(getK(), theta));
    }

    void GammaDeviate::clearCache() { _devimpl->_gamma.reset(); }

    double GammaDeviate::generate1()
    { return _devimpl->_gamma(*this->_impl->_rng); }

    std::string GammaDeviate::make_repr(bool incl_seed)
    {
        std::ostringstream oss(" ");
        oss << "galsim.GammaDeviate(";
        if (incl_seed) oss << seedstring(split(serialize(), ' ')) << ", ";
        oss << "k="<<getK()<<", ";
        oss << "theta="<<getTheta()<<")";
        return oss.str();
    }

    struct Chi2Deviate::Chi2DeviateImpl
    {
        Chi2DeviateImpl(double n) : _chi_squared(n) {}
        boost::random::chi_squared_distribution<> _chi_squared;
    };

    Chi2Deviate::Chi2Deviate(long lseed, double n) :
        BaseDeviate(lseed), _devimpl(new Chi2DeviateImpl(n)) {}

    Chi2Deviate::Chi2Deviate(const BaseDeviate& rhs, double n) :
        BaseDeviate(rhs), _devimpl(new Chi2DeviateImpl(n)) {}

    Chi2Deviate::Chi2Deviate(const Chi2Deviate& rhs) :
        BaseDeviate(rhs), _devimpl(rhs._devimpl) {}

    Chi2Deviate::Chi2Deviate(const char* str_c, double n) :
        BaseDeviate(str_c), _devimpl(new Chi2DeviateImpl(n)) {}

    double Chi2Deviate::getN() { return _devimpl->_chi_squared.n(); }

    void Chi2Deviate::setN(double n)
    {
        _devimpl->_chi_squared.param(boost::random::chi_squared_distribution<>::param_type(n));
    }

    void Chi2Deviate::clearCache() { _devimpl->_chi_squared.reset(); }

    double Chi2Deviate::generate1()
    { return _devimpl->_chi_squared(*this->_impl->_rng); }

    std::string Chi2Deviate::make_repr(bool incl_seed)
    {
        std::ostringstream oss(" ");
        oss << "galsim.Chi2Deviate(";
        if (incl_seed) oss << seedstring(split(serialize(), ' ')) << ", ";
        oss << "n="<<getN()<<")";
        return oss.str();
    }
}
