
//#define DEBUGLOGGING

#include "SBSersic.h"
#include "SBSersicImpl.h"
#include "integ/Int.h"

#ifdef DEBUGLOGGING
#include <fstream>
//std::ostream* dbgout = new std::ofstream("debug.out");
//int verbose_level = 2;
#endif

namespace galsim {

    SBSersic::SBSersic(double n, double re, double flux) : 
        SBProfile(new SBSersicImpl(n, re, flux)) {}

    SBSersic::SBSersic(const SBSersic& rhs) : SBProfile(rhs) {}

    SBSersic::~SBSersic() {}

    double SBSersic::getN() const
    { 
        assert(dynamic_cast<const SBSersicImpl*>(_pimpl.get()));
        return dynamic_cast<const SBSersicImpl&>(*_pimpl).getN(); 
    }

    double SBSersic::getHalfLightRadius() const 
    {
        assert(dynamic_cast<const SBSersicImpl*>(_pimpl.get()));
        return dynamic_cast<const SBSersicImpl&>(*_pimpl).getHalfLightRadius(); 
    }

    SBSersic::InfoBarn SBSersic::nmap;

    SBSersic::SBSersicImpl::SBSersicImpl(double n,  double re, double flux) :
        _n(n), _flux(flux), _re(re), _re_sq(_re*_re), _norm(_flux/_re_sq),
        _info(nmap.get(_n))
    {
        _ksq_max = _info->getKsqMax() / _re_sq;
        dbg<<"_ksq_max for n = "<<n<<" = "<<_ksq_max<<std::endl;
    }

    double SBSersic::SBSersicImpl::xValue(const Position<double>& p) const
    {  return _norm * _info->xValue((p.x*p.x+p.y*p.y)/_re_sq); }

    std::complex<double> SBSersic::SBSersicImpl::kValue(const Position<double>& k) const
    { 
        double ksq = k.x*k.x + k.y*k.y;
        if (ksq > _ksq_max) 
            return 0.;
        else
            return _flux * _info->kValue(ksq * _re_sq);
    }

    double SBSersic::SBSersicImpl::maxK() const { return _info->maxK() / _re; }
    double SBSersic::SBSersicImpl::stepK() const { return _info->stepK() / _re; }

    double SBSersic::SersicInfo::xValue(double xsq) const 
    { return _norm * std::exp(-_b*std::pow(xsq,_inv2n)); }

    double SBSersic::SersicInfo::kValue(double ksq) const 
    {
        // TODO: Use asymptotic formula for high-k?
        
        assert(ksq >= 0.);

        if (ksq>=_ksq_max)
            return 0.; // truncate the Fourier transform
        else if (ksq<_ksq_min)
            return 1. + ksq*(_kderiv2 + ksq*_kderiv4); // Use quartic approx at low k
        else {
            double lk=0.5*std::log(ksq); // Lookup table is logarithmic
            return _ft(lk);
        }
    }

    // Integrand class for the Hankel transform of Sersic
    class SersicIntegrand : public std::unary_function<double,double>
    {
    public:
        SersicIntegrand(double n, double b, double k):
            _invn(1./n), _b(b), _k(k) {}
        double operator()(double r) const 
        { return r*std::exp(-_b*std::pow(r, _invn))*j0(_k*r); }

    private:
        double _invn;
        double _b;
        double _k;
    };

    // Find what radius encloses (1-missing_flux_frac) of the total flux in a Sersic profile
    double SBSersic::SersicInfo::findMaxR(double missing_flux_frac, double gamma2n)
    { 
        // int(exp(-b r^1/n) r, r=R..inf) = x * int(exp(-b r^1/n) r, r=0..inf)
        //                                = x n b^-2n Gamma(2n)
        // Change variables: u = b r^1/n,
        // du = b/n r^(1-n)/n dr
        //    = b/n r^1/n dr/r
        //    = u/n dr/r
        // r dr = n du r^2 / u
        //      = n du (u/b)^2n / u
        // n b^-2n int(u^(2n-1) exp(-u), u=bR^1/n..inf) = x n b^-2n Gamma(2n)
        // Let z = b R^1/n
        //
        // int(u^(2n-1) exp(-u), u=z..inf) = x Gamma(2n)
        //
        // The lhs is an incomplete gamma function: Gamma(2n,z), which according to
        // Abramowitz & Stegun (6.5.32) has a high-z asymptotic form of:
        // Gamma(2n,z) ~= z^(2n-1) exp(-z) (1 + (2n-2)/z + (2n-2)(2n-3)/z^2 + ... )
        // ln(x Gamma(2n)) = (2n-1) ln(z) - z + 2(n-1)/z + 2(n-1)(n-2)/z^2
        // z = -ln(x Gamma(2n) + (2n-1) ln(z) + 2(n-1)/z + 2(n-1)(n-2)/z^2
        // Iterate this until it converges.  Should be quick.
        dbg<<"Find maxR for missing_flux_frac = "<<missing_flux_frac<<std::endl;
        double z0 = -std::log(missing_flux_frac * gamma2n);
        // Successive approximation method:
        double z = 4.*(_n+1.);  // A decent starting guess for a range of n.
        double oldz = 0.;
        const int MAXIT = 15;
        dbg<<"Start with z = "<<z<<std::endl;
        for(int niter=0; niter < MAXIT; ++niter) {
            oldz = z;
            z = z0 + (2.*_n-1.) * std::log(z) + 2.*(_n-1.)/z + 2.*(_n-1.)*(_n-2.)/(z*z);
            dbg<<"z = "<<z<<", dz = "<<z-oldz<<std::endl;
            if (std::abs(z-oldz) < 0.01) break;
        }
        dbg<<"Converged at z = "<<z<<std::endl;
        double R=std::pow(z/_b, _n);
        dbg<<"R = (z/b)^n = "<<R<<std::endl;
        return R;
    }

    // Constructor to initialize Sersic constants and k lookup table
    SBSersic::SersicInfo::SersicInfo(double n) : _n(n), _inv2n(1./(2.*n)) 
    {
        // Going to constrain range of allowed n to those for which testing was done
        if (_n<0.5 || _n>6.0) throw SBError("Requested Sersic index out of range");

        // Formula for b from Ciotti & Bertin (1999)
        _b = 2.*_n - (1./3.)
            + (4./405.)/_n
            + (46./25515.)/(_n*_n)
            + (131./1148175.)/(_n*_n*_n)
            - (2194697./30690717750.)/(_n*_n*_n*_n);

        double b2n = std::pow(_b,2.*_n);  // used frequently here
        double b4n = b2n*b2n;
        // The normalization factor to give unity flux integral:
        double gamma2n = tgamma(2.*_n);
        _norm = b2n / (2.*M_PI*_n*gamma2n);

        // The small-k expansion of the Hankel transform is (normalized to have flux=1):
        // 1 - Gamma(4n) / 4 b^2n Gamma(2n) + Gamma(6n) / 64 b^4n Gamma(2n)
        //   - Gamma(8n) / 2304 b^6n Gamma(2n)
        // The quadratic term of small-k expansion:
        _kderiv2 = -tgamma(4.*_n) / (4.*b2n*gamma2n); 
        // And a quartic term:
        _kderiv4 = tgamma(6.*_n) / (64.*b4n*gamma2n);

        dbg << "Building for n=" << _n << " b= " << _b << " norm= " << _norm << std::endl;
        dbg << "Deriv terms: " << _kderiv2 << " " << _kderiv4 << std::endl;

        // When is it safe to use low-k approximation?  
        // See when next term past quartic is at accuracy threshold
        double kderiv6 = tgamma(8*_n) / (2304.*b4n*b2n*gamma2n);
        dbg<<"kderiv6 = "<<kderiv6<<std::endl;
        double kmin = std::pow(sbp::kvalue_accuracy / kderiv6, 1./6.);
        dbg<<"kmin = "<<kmin<<std::endl;
        _ksq_min = kmin * kmin;

        // How far should nominal profile extend?
        // Estimate number of effective radii needed to enclose (1-alias_threshold) of flux
        double R = findMaxR(sbp::alias_threshold,gamma2n);
        // Go to at least 5 re
        if (R < 5.) R = 5.;
        dbg<<"R => "<<R<<std::endl;
        _stepK = M_PI / R;
        dbg<<"stepK = "<<_stepK<<std::endl;

        // Now start building the lookup table for FT of the profile.

        // Normalization for integral at k=0:
        double hankel_norm = _n*gamma2n/b2n;
        dbg<<"hankel_norm = "<<hankel_norm<<std::endl;

        // Keep going until at least 5 in a row have kvalues below kvalue_accuracy.
        int n_below_thresh = 0;

        double integ_maxR = findMaxR(sbp::kvalue_accuracy * hankel_norm,gamma2n);
        //double integ_maxR = integ::MOCK_INF;

        // There are two "max k" values that we care about.
        // 1) _maxK is where |f| <= maxk_threshold
        // 2) _ksq_max is where |f| <= kvalue_accuracy
        // The two thresholds are typically different, since they are used in different ways.
        // We keep track of maxlogk_1 and maxlogk_2 to keep track of each of these.
        double maxlogk_1 = 0.;
        double maxlogk_2 = 0.;
        
        double dlogk = 0.1;
        // Don't go past k = 500
        for (double logk = std::log(kmin)-0.001; logk < std::log(500.); logk += dlogk) {
            SersicIntegrand I(_n, _b, std::exp(logk));
            double val = integ::int1d(
                I, 0., integ_maxR, sbp::integration_relerr, sbp::integration_abserr*hankel_norm);
            val /= hankel_norm;
            xdbg<<"logk = "<<logk<<", ft("<<exp(logk)<<") = "<<val<<std::endl;
            _ft.addEntry(logk,val);

            if (std::abs(val) > sbp::maxk_threshold) maxlogk_1 = logk;
            if (std::abs(val) > sbp::kvalue_accuracy) maxlogk_2 = logk;

            if (std::abs(val) > sbp::kvalue_accuracy) n_below_thresh = 0;
            else ++n_below_thresh;
            if (n_below_thresh == 5) break;
        }
        // These marked the last value that didn't satisfy our requirement, so just go to 
        // the next value.
        maxlogk_1 += dlogk;
        maxlogk_2 += dlogk;
        _maxK = exp(maxlogk_1);
        xdbg<<"maxlogk_1 = "<<maxlogk_1<<std::endl;
        xdbg<<"maxK with val >= "<<sbp::maxk_threshold<<" = "<<_maxK<<std::endl;
        _ksq_max = exp(2.*maxlogk_2);
        xdbg<<"ft.argMax = "<<_ft.argMax()<<std::endl;
        xdbg<<"maxlogk_2 = "<<maxlogk_2<<std::endl;
        xdbg<<"ksq_max = "<<_ksq_max<<std::endl;

        // Next, set up the classes for photon shooting
        _radial.reset(new SersicRadialFunction(_n, _b));
        std::vector<double> range(2,0.);
        range[1] = findMaxR(sbp::shoot_flux_accuracy,gamma2n);
        _sampler.reset(new OneDimensionalDeviate( *_radial, range, true));
    }

    boost::shared_ptr<PhotonArray> SBSersic::SersicInfo::shoot(int N, UniformDeviate ud) const
    {
        dbg<<"SersicInfo shoot: N = "<<N<<std::endl;
        dbg<<"Target flux = 1.0\n";
        assert(_sampler.get());
        boost::shared_ptr<PhotonArray> result = _sampler->shoot(N,ud);
        result->scaleFlux(_norm);
        dbg<<"SersicInfo Realized flux = "<<result->getTotalFlux()<<std::endl;
        return result;
    }

    boost::shared_ptr<PhotonArray> SBSersic::SBSersicImpl::shoot(int N, UniformDeviate ud) const
    {
        dbg<<"Sersic shoot: N = "<<N<<std::endl;
        dbg<<"Target flux = "<<getFlux()<<std::endl;
        // Get photons from the SersicInfo structure, rescale flux and size for this instance
        boost::shared_ptr<PhotonArray> result = _info->shoot(N,ud);
        result->scaleFlux(_flux);
        result->scaleXY(_re);
        dbg<<"Sersic Realized flux = "<<result->getTotalFlux()<<std::endl;
        return result;
    }
}
