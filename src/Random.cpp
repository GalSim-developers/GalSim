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

#include <sys/time.h>
#include "Random.h"
#include <fcntl.h>
#include <string>
#include <vector>
#include <sstream>

namespace galsim {

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
        _rng->seed(myRandomInteger);
    }

    void BaseDeviate::seedtime()
    {
        struct timeval tp;
        gettimeofday(&tp,NULL);
        _rng->seed(tp.tv_usec);
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
            _rng->seed(alt_rng());
        }
        clearCache();
    }

    void BaseDeviate::generate(int N, double* data)
    {
        for (int i=0; i<N; ++i) data[i] = (*this)();
    }

    // Next two functions shamelessly stolen from
    // http://stackoverflow.com/questions/236129/split-a-string-in-c
    std::vector<std::string>& split(const std::string& s, char delim,
                                    std::vector<std::string>& elems) {
        std::stringstream ss(s);
        std::string item;
        while (std::getline(ss, item, delim)) {
            elems.push_back(item);
        }
        return elems;
    }

    std::vector<std::string> split(const std::string& s, char delim) {
        std::vector<std::string> elems;
        split(s, delim, elems);
        return elems;
    }

    std::string seedstring(const std::vector<std::string>& seed) {
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

    std::string UniformDeviate::make_repr(bool incl_seed)
    {
        std::ostringstream oss(" ");
        oss << "galsim.UniformDeviate(";
        if (incl_seed) oss << seedstring(split(serialize(), ' '));
        oss<<")";
        return oss.str();
    }


    std::string GaussianDeviate::make_repr(bool incl_seed)
    {
        std::ostringstream oss(" ");
        oss << "galsim.GaussianDeviate(";
        if (incl_seed) oss << seedstring(split(serialize(), ' ')) << ", ";
        oss << "mean="<<getMean()<<", ";
        oss << "sigma="<<getSigma()<<")";
        return oss.str();
    }


    std::string BinomialDeviate::make_repr(bool incl_seed)
    {
        std::ostringstream oss(" ");
        oss << "galsim.BinomialDeviate(";
        if (incl_seed) oss << seedstring(split(serialize(), ' ')) << ", ";
        oss << "N="<<getN()<<", ";
        oss << "p="<<getP()<<")";
        return oss.str();
    }


    std::string PoissonDeviate::make_repr(bool incl_seed)
    {
        std::ostringstream oss(" ");
        oss << "galsim.PoissonDeviate(";
        if (incl_seed) oss << seedstring(split(serialize(), ' ')) << ", ";
        oss << "mean="<<getMean()<<")";
        return oss.str();
    }


    std::string WeibullDeviate::make_repr(bool incl_seed)
    {
        std::ostringstream oss(" ");
        oss << "galsim.WeibullDeviate(";
        if (incl_seed) oss << seedstring(split(serialize(), ' ')) << ", ";
        oss << "a="<<getA()<<", ";
        oss << "b="<<getB()<<")";
        return oss.str();
    }


    std::string GammaDeviate::make_repr(bool incl_seed)
    {
        std::ostringstream oss(" ");
        oss << "galsim.GammaDeviate(";
        if (incl_seed) oss << seedstring(split(serialize(), ' ')) << ", ";
        oss << "k="<<getK()<<", ";
        oss << "theta="<<getTheta()<<")";
        return oss.str();
    }


    std::string Chi2Deviate::make_repr(bool incl_seed)
    {
        std::ostringstream oss(" ");
        oss << "galsim.Chi2Deviate(";
        if (incl_seed) oss << seedstring(split(serialize(), ' ')) << ", ";
        oss << "n="<<getN()<<")";
        return oss.str();
    }

    void PoissonDeviate::setMean(double mean)
    {
        // Near 2**31, the boost poisson rng can wrap around to negative integers, which
        // is bad.  But this high, the Gaussian approximation is extremely accurate, so
        // just use that.
        const double MAX_POISSON = 1<<30;

        if (mean == getMean()) return;
        _pd.param(boost::random::poisson_distribution<>::param_type(mean));
        if (mean > MAX_POISSON) {
            if (!_gd) {
                _gd.reset(new boost::random::normal_distribution<>(mean, std::sqrt(mean)));
            } else {
                _gd->param(boost::random::normal_distribution<>::param_type(mean, std::sqrt(mean)));
            }
            _getValue = &PoissonDeviate::getGDValue;
        } else {
            _gd.reset();
            _getValue = &PoissonDeviate::getPDValue;
        }
    }

    double PoissonDeviate::_val()
    {
        return (this->*_getValue)();
    }

    double PoissonDeviate::getPDValue()
    {
        return _pd(*this->_rng);
    }

    double PoissonDeviate::getGDValue()
    {
        return (*_gd)(*this->_rng);
    }

}
