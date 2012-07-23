
//#define DEBUGLOGGING

#include "SBKolmogorov.h"
#include "SBKolmogorovImpl.h"

#ifdef DEBUGLOGGING
#include <fstream>
std::ostream* dbgout = new std::ofstream("debug.out");
int verbose_level = 2;
#endif

namespace galsim {

    // A static variable for the SBKolmogorov class:
    KolmogorovInfo SBKolmogorov::SBKolmogorovImpl::_info;

    SBKolmogorov::SBKolmogorov(double lam_over_r0, double flux) :
        SBProfile(new SBKolmogorovImpl(lam_over_r0, flux)) {}

    SBKolmogorov::SBKolmogorov(const SBKolmogorov& rhs) : SBProfile(rhs) {}

    SBKolmogorov::~SBKolmogorov() {}

    double SBKolmogorov::getLamOverR0() const 
    {
        assert(dynamic_cast<const SBKolmogorovImpl*>(_pimpl.get()));
        return dynamic_cast<const SBKolmogorovImpl&>(*_pimpl).getLamOverR0(); 
    }

    // The "magic" number 2.992934 below comes from the standard form of the Kolmogorov spectrum
    // from Racine, 1996 PASP, 108, 699 (who in turn is quoting Fried, 1966, JOSA, 56, 1372):
    // T(k) = exp(-1/2 D(k)) 
    // D(k) = 6.8839 (lambda/r0 k/2Pi)^(5/3)
    //
    // We convert this into T(k) = exp(-(k/k0)^5/3) for efficiency,
    // which implies 1/2 6.8839 (lambda/r0 / 2Pi)^5/3 = (1/k0)^5/3
    // k0 * lambda/r0 = 2Pi * (6.8839 / 2)^-3/5 = 2.992934
    //
    SBKolmogorov::SBKolmogorovImpl::SBKolmogorovImpl(double lam_over_r0, double flux) :
        _lam_over_r0(lam_over_r0), 
        _k0(2.992934 / lam_over_r0), 
        _flux(flux), 
        _k0sq(_k0*_k0),
        _inv_k0sq(1./_k0sq),
        _xnorm(1.)  // TODO: This isn't right need to figure out what the right norm is.
    {}

    double SBKolmogorov::SBKolmogorovImpl::xValue(const Position<double>& p) const 
    {
        double r = sqrt(p.x*p.x+p.y*p.y) * _k0;
        return _xnorm * _info.xValue(r);
    }

    double KolmogorovInfo::xValue(double r) const 
    { return r < _radial.argMax() ? _radial(r) : 0.; }

    std::complex<double> SBKolmogorov::SBKolmogorovImpl::kValue(const Position<double>& k) const
    {
        double ksq = (k.x*k.x+k.y*k.y) * _inv_k0sq;
        return _flux * _info.kValue(ksq);
    }

    // Set maxK to where kValue drops to maxk_threshold
    double SBKolmogorov::SBKolmogorovImpl::maxK() const 
    { return _info.maxK() * _k0; }

    // The amount of flux missed in a circle of radius pi/stepk should miss at 
    // most alias_threshold of the flux.
    double SBKolmogorov::SBKolmogorovImpl::stepK() const
    { return _info.stepK() * _k0; }

    // f(k) = exp(-(k/k0)^5/3)
    // The input value should already be (k/k0)^2
    double KolmogorovInfo::kValue(double ksq) const 
    { return exp(-std::pow(ksq,5./6.)); }

    // Integrand class for the Hankel transform of Kolmogorov
    class KolmogorovIntegrand : public std::unary_function<double,double>
    {
    public:
        KolmogorovIntegrand(double r) : _r(r) {}
        double operator()(double k) const
        { return k*std::exp(-std::pow(k, 5./3.))*j0(k*_r); }

    private:
        double _r;
    };
     
    // Constructor to initialize Kolmogorov constants and xvalue lookup table
    KolmogorovInfo::KolmogorovInfo() : _radial(TableDD::spline)
    {
        dbg<<"Initializing KolmogorovInfo\n";

        // Calculate maxK:
        // exp(-k^5/3) = kvalue_accuracy
        _maxk = std::pow(-std::log(sbp::kvalue_accuracy),3./5.);
        dbg<<"maxK = "<<_maxk<<std::endl;

        double integ_maxR = integ::MOCK_INF;
        // Build the table for the radial function.
        double dr = 0.1;
        // Start with f(0), which is analytic:
        // According to Wolfram Alpha:
        // Integrate[k*exp(-k^5/3),{k,0,infinity}] = 1/5 3^(2/5) Gamma(2/5)
        //    = 0.68844821404369641022575576988...
        double prev = 0.68844821404369641022575576988;
        _radial.addEntry(0.,prev);
        // Along the way accumulate the flux integral to determine the radius
        // that encloses (1-alias_threshold) of the flux.
        double sum = 0.;
        double thresh = (1.-sbp::alias_threshold) / (2.*M_PI*dr);
        double R = 0.;
        // Continue until at least 5 in a row with f(r) < xvalue_threshold
        int n_below_thresh = 0;
        for (double r = dr; ; r += dr) {
            KolmogorovIntegrand I(r);
            double val = integ::int1d(
                I, 0., integ_maxR, sbp::integration_relerr, sbp::integration_abserr);
            xdbg<<"f("<<r<<") = "<<val<<std::endl;
            _radial.addEntry(r,val);

            // Accumulate int(r*f(r)) / dr  (i.e. don't include 2*pi*dr factors as part of sum)
            sum += r * val;
            xdbg<<"sum -> "<<sum<<std::endl;
            if (R == 0. && sum > thresh) R = r;

            if (std::abs(val) > sbp::xvalue_accuracy) n_below_thresh = 0;
            else ++n_below_thresh;
            if (n_below_thresh == 5) break;
            prev = val;
        }
        dbg<<"Done loop to build radial function.\n";
        dbg<<"R = "<<R<<std::endl;
        _stepk = M_PI/R;
        dbg<<"stepK = "<<_stepk<<std::endl;
        dbg<<"sum = "<<sum<<std::endl;
        dbg<<"sum * 2pi * dr = "<<sum*2.*M_PI*dr<<std::endl;

        // Next, set up the sampler for photon shooting
        std::vector<double> range(2,0.);
        range[1] = _radial.argMax();
        _sampler.reset(new OneDimensionalDeviate(_radial, range, true));
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
        boost::shared_ptr<PhotonArray> result = _info.shoot(N,ud);
        result->scaleFlux(_flux);
        result->scaleXY(1./_k0);
        dbg<<"Kolmogorov Realized flux = "<<result->getTotalFlux()<<std::endl;
        return result;
    }
}
