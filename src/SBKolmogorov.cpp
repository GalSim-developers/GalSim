// -*- c++ -*-
/*
 * Copyright 2012, 2013 The GalSim developers:
 * https://github.com/GalSim-developers
 *
 * This file is part of GalSim: The modular galaxy image simulation toolkit.
 *
 * GalSim is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * GalSim is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with GalSim.  If not, see <http://www.gnu.org/licenses/>
 */

//#define DEBUGLOGGING

#include "SBKolmogorov.h"
#include "SBKolmogorovImpl.h"

#ifdef DEBUGLOGGING
#include <fstream>
//std::ostream* dbgout = new std::ofstream("debug.out");
//int verbose_level = 2;
#endif

// Uncomment this to do the calculation that solves for the conversion between lam_over_r0
// and fwhm and hlr.
// (Solved values are put into Kolmogorov class in galsim/base.py = 0.975865, 0.554811)
//#define SOLVE_FWHM_HLR

#ifdef SOLVE_FWHM_HLR
#include "Solve.h"
#endif

namespace galsim {

    SBKolmogorov::SBKolmogorov(double lam_over_r0, double flux,
                               boost::shared_ptr<GSParams> gsparams) :
        SBProfile(new SBKolmogorovImpl(lam_over_r0, flux, gsparams)) {}

    SBKolmogorov::SBKolmogorov(const SBKolmogorov& rhs) : SBProfile(rhs) {}

    SBKolmogorov::~SBKolmogorov() {}

    double SBKolmogorov::getLamOverR0() const 
    {
        assert(dynamic_cast<const SBKolmogorovImpl*>(_pimpl.get()));
        return static_cast<const SBKolmogorovImpl&>(*_pimpl).getLamOverR0(); 
    }

    const int MAX_KOLMOGOROV_INFO = 100;

    LRUCache<const GSParams*, KolmogorovInfo>
        SBKolmogorov::SBKolmogorovImpl::cache(MAX_KOLMOGOROV_INFO);

    // The "magic" number 2.992934 below comes from the standard form of the Kolmogorov spectrum
    // from Racine, 1996 PASP, 108, 699 (who in turn is quoting Fried, 1966, JOSA, 56, 1372):
    // T(k) = exp(-1/2 D(k)) 
    // D(k) = 6.8839 (lambda/r0 k/2Pi)^(5/3)
    //
    // We convert this into T(k) = exp(-(k/k0)^5/3) for efficiency,
    // which implies 1/2 6.8839 (lambda/r0 / 2Pi)^5/3 = (1/k0)^5/3
    // k0 * lambda/r0 = 2Pi * (6.8839 / 2)^-3/5 = 2.992934
    //
    SBKolmogorov::SBKolmogorovImpl::SBKolmogorovImpl(
        double lam_over_r0, double flux,
        boost::shared_ptr<GSParams> gsparams) :
        SBProfileImpl(gsparams),
        _lam_over_r0(lam_over_r0), 
        _k0(2.992934 / lam_over_r0), 
        _k0sq(_k0*_k0),
        _inv_k0(1./_k0),
        _inv_k0sq(1./_k0sq),
        _flux(flux), 
        _xnorm(_flux * _k0sq),
        _info(cache.get(this->gsparams.get()))
    {
        dbg<<"SBKolmogorov:\n";
        dbg<<"lam_over_r0 = "<<_lam_over_r0<<std::endl;
        dbg<<"k0 = "<<_k0<<std::endl;
        dbg<<"flux = "<<_flux<<std::endl;
        dbg<<"xnorm = "<<_xnorm<<std::endl;
    }

    double SBKolmogorov::SBKolmogorovImpl::xValue(const Position<double>& p) const 
    {
        double r = sqrt(p.x*p.x+p.y*p.y) * _k0;
        return _xnorm * _info->xValue(r);
    }

    double KolmogorovInfo::xValue(double r) const 
    { return r < _radial.argMax() ? _radial(r) : 0.; }

    std::complex<double> SBKolmogorov::SBKolmogorovImpl::kValue(const Position<double>& k) const
    {
        double ksq = (k.x*k.x+k.y*k.y) * _inv_k0sq;
        return _flux * _info->kValue(ksq);
    }

    void SBKolmogorov::SBKolmogorovImpl::fillXValue(tmv::MatrixView<double> val,
                                                    double x0, double dx, int ix_zero,
                                                    double y0, double dy, int iy_zero) const
    {
        dbg<<"SBKolmogorov fillXValue\n";
        dbg<<"x = "<<x0<<" + ix * "<<dx<<", ix_zero = "<<ix_zero<<std::endl;
        dbg<<"y = "<<y0<<" + iy * "<<dy<<", iy_zero = "<<iy_zero<<std::endl;
        if (ix_zero != 0 || iy_zero != 0) {
            xdbg<<"Use Quadrant\n";
            fillXValueQuadrant(val,x0,dx,ix_zero,y0,dy,iy_zero);
        } else {
            xdbg<<"Non-Quadrant\n";
            assert(val.stepi() == 1);
            const int m = val.colsize();
            const int n = val.rowsize();
            typedef tmv::VIt<double,1,tmv::NonConj> It;

            x0 *= _k0;
            dx *= _k0;
            y0 *= _k0;
            dy *= _k0;

            for (int j=0;j<n;++j,y0+=dy) {
                double x = x0;
                double ysq = y0*y0;
                It valit = val.col(j).begin();
                for (int i=0;i<m;++i,x+=dx) {
                    double r = sqrt(x*x + ysq);
                    *valit++ = _xnorm * _info->xValue(r);
                }
            }
        }
    }

    void SBKolmogorov::SBKolmogorovImpl::fillKValue(tmv::MatrixView<std::complex<double> > val,
                                                    double x0, double dx, int ix_zero,
                                                    double y0, double dy, int iy_zero) const
    {
        dbg<<"SBKolmogorov fillKValue\n";
        dbg<<"x = "<<x0<<" + ix * "<<dx<<", ix_zero = "<<ix_zero<<std::endl;
        dbg<<"y = "<<y0<<" + iy * "<<dy<<", iy_zero = "<<iy_zero<<std::endl;
        if (ix_zero != 0 || iy_zero != 0) {
            xdbg<<"Use Quadrant\n";
            fillKValueQuadrant(val,x0,dx,ix_zero,y0,dy,iy_zero);
        } else {
            xdbg<<"Non-Quadrant\n";
            assert(val.stepi() == 1);
            const int m = val.colsize();
            const int n = val.rowsize();
            typedef tmv::VIt<std::complex<double>,1,tmv::NonConj> It;

            x0 *= _inv_k0;
            dx *= _inv_k0;
            y0 *= _inv_k0;
            dy *= _inv_k0;

            for (int j=0;j<n;++j,y0+=dy) {
                double x = x0;
                double ysq = y0*y0;
                It valit(val.col(j).begin().getP(),1);
                for (int i=0;i<m;++i,x+=dx) *valit++ = _flux * _info->kValue(x*x + ysq);
            }
        }
    }

    void SBKolmogorov::SBKolmogorovImpl::fillXValue(tmv::MatrixView<double> val,
                                                    double x0, double dx, double dxy,
                                                    double y0, double dy, double dyx) const
    {
        dbg<<"SBKolmogorov fillXValue\n";
        dbg<<"x = "<<x0<<" + ix * "<<dx<<" + iy * "<<dxy<<std::endl;
        dbg<<"y = "<<y0<<" + ix * "<<dyx<<" + iy * "<<dy<<std::endl;
        assert(val.stepi() == 1);
        assert(val.canLinearize());
        const int m = val.colsize();
        const int n = val.rowsize();
        typedef tmv::VIt<double,1,tmv::NonConj> It;

        x0 *= _k0;
        dx *= _k0;
        dxy *= _k0;
        y0 *= _k0;
        dy *= _k0;
        dyx *= _k0;

        It valit = val.linearView().begin();
        for (int j=0;j<n;++j,x0+=dxy,y0+=dy) {
            double x = x0;
            double y = y0;
            It valit = val.col(j).begin();
            for (int i=0;i<m;++i,x+=dx,y+=dyx) {
                double r = sqrt(x*x + y*y);
                *valit++ = _xnorm * _info->xValue(r);
            }
        }
    }

    void SBKolmogorov::SBKolmogorovImpl::fillKValue(tmv::MatrixView<std::complex<double> > val,
                                                    double x0, double dx, double dxy,
                                                    double y0, double dy, double dyx) const
    {
        dbg<<"SBKolmogorov fillKValue\n";
        dbg<<"x = "<<x0<<" + ix * "<<dx<<" + iy * "<<dxy<<std::endl;
        dbg<<"y = "<<y0<<" + ix * "<<dyx<<" + iy * "<<dy<<std::endl;
        assert(val.stepi() == 1);
        assert(val.canLinearize());
        const int m = val.colsize();
        const int n = val.rowsize();
        typedef tmv::VIt<std::complex<double>,1,tmv::NonConj> It;

        x0 *= _inv_k0;
        dx *= _inv_k0;
        dxy *= _inv_k0;
        y0 *= _inv_k0;
        dy *= _inv_k0;
        dyx *= _inv_k0;

        It valit(val.linearView().begin().getP(),1);
        for (int j=0;j<n;++j,x0+=dxy,y0+=dy) {
            double x = x0;
            double y = y0;
            It valit(val.col(j).begin().getP(),1);
            for (int i=0;i<m;++i,x+=dx,y+=dyx) *valit++ = _flux * _info->kValue(x*x+y*y);
        }
    }

    // Set maxK to where kValue drops to maxk_threshold
    double SBKolmogorov::SBKolmogorovImpl::maxK() const 
    { return _info->maxK() * _k0; }

    // The amount of flux missed in a circle of radius pi/stepk should be at 
    // most alias_threshold of the flux.
    double SBKolmogorov::SBKolmogorovImpl::stepK() const
    { return _info->stepK() * _k0; }

    // f(k) = exp(-(k/k0)^5/3)
    // The input value should already be (k/k0)^2
    double KolmogorovInfo::kValue(double ksq) const 
    { return exp(-std::pow(ksq,5./6.)); }

    // Integrand class for the Hankel transform of Kolmogorov
    class KolmIntegrand : public std::unary_function<double,double>
    {
    public:
        KolmIntegrand(double r) : _r(r) {}
        double operator()(double k) const
        { return k*std::exp(-std::pow(k, 5./3.))*j0(k*_r); }

    private:
        double _r;
    };

    // Perform the integral
    class KolmXValue : public std::unary_function<double,double>
    {
    public:
        KolmXValue(const GSParams* gsparams) : 
            _gsparams(gsparams) {}

        double operator()(double r) const
        { 
            const double integ_maxK = integ::MOCK_INF;
            KolmIntegrand I(r);
            return integ::int1d(I, 0., integ_maxK,
                                _gsparams->integration_relerr,
                                _gsparams->integration_abserr);
        }
    private:
        const GSParams* _gsparams;
    };

#ifdef SOLVE_FWHM_HLR
    // XValue - target  (used for solving for fwhm)
    class KolmTargetValue : public std::unary_function<double,double>
    {
    public:
        KolmTargetValue(double target, const GSParams* gsparams) : _target(target,gsparams) {}
        double operator()(double r) const { return f(r) - _target; }
    private:
        KolmXValue f;
        double _target;
    };

    class KolmXValueTimes2piR : public std::unary_function<double,double>
    {
    public:
        KolmXValueTimes2piR(const GSParams* gsparams) : f(gsparams) {}

        double operator()(double r) const
        { return f(r) * r; }
    private:
        KolmXValue f;
    };

    class KolmEnclosedFlux : public std::unary_function<double,double>
    {
    public:
        KolmEnclosedFlux(const GSParams* gsparams) : f(gsparams), _gsparams(gsparams) {}
        double operator()(double r) const 
        {
            return integ::int1d(f, 0., r,
                                _gsparams->integration_relerr,
                                _gsparams->integration_abserr);
        }
    private:
        KolmXValueTimes2piR f;
        const GSParams* _gsparams;
    };

    class KolmTargetFlux : public std::unary_function<double,double>
    {
    public:
        KolmTargetFlux(double target, const GSParams* gsparams) : f(gsparams), _target(target) {}
        double operator()(double r) const { return f(r) - _target; }
    private:
        KolmEnclosedFlux f;
        double _target;
    };
#endif

    // Constructor to initialize Kolmogorov constants and xvalue lookup table
    KolmogorovInfo::KolmogorovInfo(const GSParams* gsparams) : _radial(TableDD::spline)
    {
        dbg<<"Initializing KolmogorovInfo\n";

        // Calculate maxK:
        // exp(-k^5/3) = kvalue_accuracy
        _maxk = std::pow(-std::log(gsparams->kvalue_accuracy),3./5.);
        dbg<<"maxK = "<<_maxk<<std::endl;

        // Build the table for the radial function.
        double dr = 0.5/_maxk;
        // Start with f(0), which is analytic:
        // According to Wolfram Alpha:
        // Integrate[k*exp(-k^5/3),{k,0,infinity}] = 3/5 Gamma(6/5)
        //    = 0.55090124543985636638457099311149824;
        double val = 0.55090124543985636638457099311149824 / (2.*M_PI);
        _radial.addEntry(0.,val);
        xdbg<<"f(0) = "<<val<<std::endl;
        // Along the way accumulate the flux integral to determine the radius
        // that encloses (1-alias_threshold) of the flux.
        double sum = 0.;
        double thresh1 = (1.-gsparams->alias_threshold) / (2.*M_PI*dr);
        double thresh2 = 0.999 / (2.*M_PI*dr);
        double R = 0.;
        // Continue until accumulate 0.999 of the flux
        KolmXValue xval_func(gsparams);
        for (double r = dr; sum < thresh2; r += dr) {
            val = xval_func(r) / (2.*M_PI);
            xdbg<<"f("<<r<<") = "<<val<<std::endl;
            _radial.addEntry(r,val);

            // Accumulate int(r*f(r)) / dr  (i.e. don't include 2*pi*dr factor as part of sum)
            sum += r * val;
            xdbg<<"sum = "<<sum<<"  thresh1 = "<<thresh1<<"  thesh2 = "<<thresh2<<std::endl;
            xdbg<<"sum*2*pi*dr "<<sum*2.*M_PI*dr<<std::endl;
            if (R == 0. && sum > thresh1) R = r;
        }
        dbg<<"Done loop to build radial function.\n";
        dbg<<"R = "<<R<<std::endl;
        _stepk = M_PI/R;
        dbg<<"stepk = "<<_stepk<<std::endl;
        dbg<<"sum*2*pi*dr = "<<sum*2.*M_PI*dr<<"   (should ~= 0.999)\n";

        // Next, set up the sampler for photon shooting
        std::vector<double> range(2,0.);
        range[1] = _radial.argMax();
        _sampler.reset(new OneDimensionalDeviate(_radial, range, true));

#ifdef SOLVE_FWHM_HLR
        // Improve upon the conversion between lam_over_r0 and fwhm:
        KolmTargetValue fwhm_func(0.55090124543985636638457099311149824 / 2., gsparams);
        double r1 = 1.4;
        double r2 = 1.5;
        Solve<KolmTargetValue> fwhm_solver(fwhm_func,r1,r2);
        fwhm_solver.setMethod(Brent);
        double rd = fwhm_solver.root();
        xdbg<<"Root is "<<rd<<std::endl;
        // This is in units of 1/k0.  k0 = 2.992934 / lam_over_r0
        // It's also the half-width hal-max, so * 2 to get fwhm.
        xdbg<<"fwhm = "<<rd * 2. / 2.992934<<" * lam_over_r0\n";

        // Confirm that flux function gets unit flux when integrated to infinity:
        KolmEnclosedFlux enc_flux;
        for(double rmax = 0.; rmax < 20.; rmax += 1.) {
            dbg<<"Flux enclosed by r="<<rmax<<" = "<<enc_flux(rmax)<<std::endl;
        }

        // Next find the conversion between lam_over_r0 and hlr:
        KolmTargetFlux hlr_func(0.5);
        r1 = 1.6;
        r2 = 1.7;
        Solve<KolmTargetFlux> hlr_solver(hlr_func,r1,r2);
        hlr_solver.setMethod(Brent);
        rd = hlr_solver.root();
        xdbg<<"Root is "<<rd<<std::endl;
        dbg<<"Flux enclosed by r="<<rd<<" = "<<enc_flux(rd)<<std::endl;
        // This is in units of 1/k0.  k0 = 2.992934 / lam_over_r0
        xdbg<<"hlr = "<<rd / 2.992934<<" * lam_over_r0\n";
#endif
    }

    boost::shared_ptr<PhotonArray> KolmogorovInfo::shoot(int N, UniformDeviate ud) const
    {
        dbg<<"KolmogorovInfo shoot: N = "<<N<<std::endl;
        dbg<<"Target flux = 1.0\n";
        assert(_sampler.get());
        boost::shared_ptr<PhotonArray> result = _sampler->shoot(N,ud);
        //result->scaleFlux(_norm);
        dbg<<"KolmogorovInfo Realized flux = "<<result->getTotalFlux()<<std::endl;
        return result;
    }

    boost::shared_ptr<PhotonArray> SBKolmogorov::SBKolmogorovImpl::shoot(
        int N, UniformDeviate ud) const
    {
        dbg<<"Kolmogorov shoot: N = "<<N<<std::endl;
        dbg<<"Target flux = "<<getFlux()<<std::endl;
        // Get photons from the KolmogorovInfo structure, rescale flux and size for this instance
        boost::shared_ptr<PhotonArray> result = _info->shoot(N,ud);
        result->scaleFlux(_flux);
        result->scaleXY(1./_k0);
        dbg<<"Kolmogorov Realized flux = "<<result->getTotalFlux()<<std::endl;
        return result;
    }
}
