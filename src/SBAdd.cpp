
//#define DEBUGLOGGING

#include "SBAdd.h"
#include "SBAddImpl.h"

#ifdef DEBUGLOGGING
#include <fstream>
//std::ostream* dbgout = new std::ofstream("debug.out");
//int verbose_level = 2;
#endif

namespace galsim {

    SBAdd::SBAdd(const SBProfile& s1, const SBProfile& s2) :
        SBProfile(new SBAddImpl(s1,s2)) {}

    SBAdd::SBAdd(const std::list<SBProfile>& slist) :
        SBProfile(new SBAddImpl(slist)) {}

    SBAdd::SBAdd(const SBAdd& rhs) : SBProfile(rhs) {}
    
    SBAdd::~SBAdd() {}

    void SBAdd::SBAddImpl::add(const SBProfile& rhs)
    {
        xdbg<<"Start SBAdd::add.  Adding item # "<<_plist.size()+1<<std::endl;
        // Add new summand(s) to the _plist:
        assert(SBProfile::GetImpl(rhs));
        const SBAddImpl *sba = dynamic_cast<const SBAddImpl*>(SBProfile::GetImpl(rhs));
        if (sba) {
            // If rhs is an SBAdd, copy its full list here
            _plist.insert(_plist.end(),sba->_plist.begin(),sba->_plist.end());
        } else {
            _plist.push_back(rhs);
        }
    }

    void SBAdd::SBAddImpl::initialize() 
    {
        _sumflux = _sumfx = _sumfy = 0.;
        _maxMaxK = _minStepK = 0.;
        _allAxisymmetric = _allAnalyticX = _allAnalyticK = true;
        _anyHardEdges = false;

        // Accumulate properties of all summands
        for(ConstIter it=_plist.begin(); it!=_plist.end(); ++it) {
            xdbg<<"SBAdd component has maxK, stepK = "<<
                it->maxK()<<" , "<<it->stepK()<<std::endl;
            _sumflux += it->getFlux();
            _sumfx += it->getFlux() * it->centroid().x;
            _sumfy += it->getFlux() * it->centroid().x;
            if ( it->maxK() > _maxMaxK) 
                _maxMaxK = it->maxK();
            if ( _minStepK<=0. || (it->stepK() < _minStepK) ) 
                _minStepK = it->stepK();
            _allAxisymmetric = _allAxisymmetric && it->isAxisymmetric();
            _anyHardEdges = _anyHardEdges || it->hasHardEdges();
            _allAnalyticX = _allAnalyticX && it->isAnalyticX();
            _allAnalyticK = _allAnalyticK && it->isAnalyticK();
        }
        xdbg<<"Net maxK, stepK = "<<_maxMaxK<<" , "<<_minStepK<<std::endl;
    }

    double SBAdd::SBAddImpl::xValue(const Position<double>& p) const 
    {
        ConstIter pptr = _plist.begin();
        assert(pptr != _plist.end());
        double xv = pptr->xValue(p);
        for (++pptr; pptr != _plist.end(); ++pptr)
            xv += pptr->xValue(p);
        return xv;
    } 

    std::complex<double> SBAdd::SBAddImpl::kValue(const Position<double>& k) const 
    {
        ConstIter pptr = _plist.begin();
        assert(pptr != _plist.end());
        std::complex<double> kv = pptr->kValue(k);
        for (++pptr; pptr != _plist.end(); ++pptr)
            kv += pptr->kValue(k);
        return kv;
    } 

    void SBAdd::SBAddImpl::fillKGrid(KTable& kt) const 
    {
        if (_plist.empty()) kt.clear();
        ConstIter pptr = _plist.begin();
        assert(SBProfile::GetImpl(*pptr));
        SBProfile::GetImpl(*pptr)->fillKGrid(kt);
        if (++pptr != _plist.end()) {
            KTable k2(kt.getN(),kt.getDk());
            for ( ; pptr!= _plist.end(); ++pptr) {
                assert(SBProfile::GetImpl(*pptr));
                SBProfile::GetImpl(*pptr)->fillKGrid(k2);
                kt.accumulate(k2);
            }
        }
    }

    void SBAdd::SBAddImpl::fillXGrid(XTable& xt) const 
    {
        if (_plist.empty()) xt.clear();
        ConstIter pptr = _plist.begin();
        assert(SBProfile::GetImpl(*pptr));
        SBProfile::GetImpl(*pptr)->fillXGrid(xt);
        if (++pptr != _plist.end()) {
            XTable x2(xt.getN(),xt.getDx());
            for ( ; pptr!= _plist.end(); ++pptr) {
                assert(SBProfile::GetImpl(*pptr));
                SBProfile::GetImpl(*pptr)->fillXGrid(x2);
                xt.accumulate(x2);
            }
        }
    }

    double SBAdd::SBAddImpl::getPositiveFlux() const 
    {
        double result = 0.;
        for (ConstIter pptr = _plist.begin(); pptr != _plist.end(); ++pptr) {
            result += pptr->getPositiveFlux();  
        }
        return result;
    }

    double SBAdd::SBAddImpl::getNegativeFlux() const 
    {
        double result = 0.;
        for (ConstIter pptr = _plist.begin(); pptr != _plist.end(); ++pptr) {
            result += pptr->getNegativeFlux();  
        }
        return result;
    }

    boost::shared_ptr<PhotonArray> SBAdd::SBAddImpl::shoot(int N, UniformDeviate u) const 
    {
        dbg<<"Add shoot: N = "<<N<<std::endl;
        dbg<<"Target flux = "<<getFlux()<<std::endl;
        double totalAbsoluteFlux = getPositiveFlux() + getNegativeFlux();
        double fluxPerPhoton = totalAbsoluteFlux / N;

        // Initialize the output array
        boost::shared_ptr<PhotonArray> result(new PhotonArray(0));

        double remainingAbsoluteFlux = totalAbsoluteFlux;
        int remainingN = N;

        // Get photons from each summand, using BinomialDeviate to
        // randomize distribution of photons among summands
        for (ConstIter pptr = _plist.begin(); pptr!= _plist.end(); ++pptr) {
            double thisAbsoluteFlux = pptr->getPositiveFlux() + pptr->getNegativeFlux();

            // How many photons to shoot from this summand?
            int thisN = remainingN;  // All of what's left, if this is the last summand...
            std::list<SBProfile>::const_iterator nextPtr = pptr;
            ++nextPtr;
            if (nextPtr!=_plist.end()) {
                // otherwise allocate a randomized fraction of the remaining photons to this summand:
                BinomialDeviate bd(u, remainingN, thisAbsoluteFlux/remainingAbsoluteFlux);
                thisN = bd();
            }
            if (thisN > 0) {
                boost::shared_ptr<PhotonArray> thisPA = pptr->shoot(thisN, u);
                // Now rescale the photon fluxes so that they are each nominally fluxPerPhoton
                // whereas the shoot() routine would have made them each nominally 
                // thisAbsoluteFlux/thisN
                thisPA->scaleFlux(fluxPerPhoton*thisN/thisAbsoluteFlux);
                result->append(*thisPA);
            }
            remainingN -= thisN;
            remainingAbsoluteFlux -= thisAbsoluteFlux;
            if (remainingN <=0) break;
            if (remainingAbsoluteFlux <= 0.) break;
        }
        
        dbg<<"Add Realized flux = "<<result->getTotalFlux()<<std::endl;

        // This process produces correlated photons, so mark the resulting array as such.
        if (_plist.size() > 1) result->setCorrelated();
        
        return result;
    }
}
