// -*- c++ -*-
#ifndef IMAGENOISE_H
#define IMAGENOISE_H

/** 
 * @file CCDNoise.h @brief Add noise to image using standard CCD model
 *
 */

#include <cmath>
#include "Std.h"
#include "Random.h"
#include "Image.h"

namespace galsim {
    /** 
     * @brief Class implementing basic CCD noise model.  
     *
     * CCDNoise instance is given a "read noise" and gain level.  With these parameters set,
     * it can operate on an Image by adding noise to each pixel.  The noise has two
     * components: first, a Poisson deviate with variance equal to ( max(pixel value, 0) / gain);
     * second, Gaussian noise with RMS value of (readNoise / gain).  The class must
     * be given a reference to a UniformDeviate when constructed, which will be the source
     * of random values for the noise implementation.
     * readNoise=0 will shut off the Gaussian noise.
     * gain<=0 will shut off the Poisson noise, and Gaussian value will just have value RMS=readNoise.
     */
    class CCDNoise {
    private: 
        double gain;    // flux corresponding to one photon
        double readNoise; // std. dev. of uniform Gaussian noise (when divided by gain).
        UniformDeviate& ud;
        // Gaussian and Poisson deviates will use common uniform deviate generator
        GaussianDeviate gd;
        PoissonDeviate pd;
    public:
         /**
         * @brief Construct a new noise model
         *
         * @param[in] ud_ UniformDeviate that will be called to generate all randoms
         * @param[in] gain_ Electrons per ADU in the input Images, used for Poisson noise.
         * @param[in] readNoise_ RMS of Gaussian noise, in electrons (if gain>0.) or ADU (gain<=0.)
         */
       CCDNoise(UniformDeviate& ud_, double gain_=1., double readNoise_=0.):
            gain(gain_),
            readNoise(readNoise_),
            ud(ud_), 
            gd(ud, 0., 1.),
            pd(ud) {
            if (readNoise > 0.) gd.setMean( readNoise / (gain > 0. ? gain : 1.));
        }
        /**
         * @brief Report current gain value
         *
         * @return Gain value (e/ADU)
         */
        double getGain() const {return gain;}
        /**
         * @brief Report current read noise
         *
         * @return Read noise value (e, if gain>0, else in ADU)
         */
        double getReadNoise() const {return readNoise;}
        /**
         * @brief Set gain value
         *
         * @param[in] gain_ Gain value (e/ADU)
         */
        double setGain(double gain_) const {
            gain = gain_;
            if (readNoise > 0.) gd.setMean( readNoise / (gain > 0. ? gain : 1.));
        }
        /**
         * @brief Set read noise
         *
         * @param[in] readNoise_ Read noise value (e, if gain>0, else in ADU)
         */
        double setReadNoise(double readNoise_) const {
            readNoise = readNoise_;
            if (readNoise > 0.) gd.setMean( readNoise / (gain > 0. ? gain : 1.));
        }

        /**
         * @brief Add noise to an Image
         *
         * Poisson and/or Gaussian noise are added to each pixel of the image according
         * to standard CCD model.
         * @param[in,out] data The Image to be noise-ified.
         */
        void operator() (Image<> data) {
            // Above this many e's, assume Poisson distribution =Gaussian 
            static const double MAX_POISSON=1.e5;
            double dx;
            if (!data.getHdrValue("DX", dx)) dx=1.;
            double pixsig = sigma*dx;
            double sigsq = pixsig*pixsig;
            Image<> invvar(data.getBounds());

            for (int y = data.YMin(); y <= data.YMax(); y++) {  // iterate over y
                Image<>::iter ee=data.rowEnd(y);
                Image<>::iter it;  // for data
                Image<>::iter it2; // for invvar
                for (it=data.rowBegin(y), it2=invvar.rowBegin(y);  // iterate over x
                     it!=ee;
                     ++it, ++it2) {
                    double var = 0.;
                    double f=*it;
                    if (f0>0. && f>0.) {
                        var = f0*f;
                        double ne = f/f0;
                        if (ne>MAX_POISSON) *it = floor(ne+sqrt(ne)*g+0.5)*f0;
                        else {p.Reset(ne); *it=p*f0;}
                    }
                    if (sigma>0.) {
                        var += sigsq;
                        *it += pixsig*g;
                    }
                    *it2 = 1./var;
                } 
            }
            return invvar;
        }
        /**
         * @brief Add noise to an Image and also report variance of each pixel.
         *
         * Adds noise as in operator()(Image) signature, but second Image is filled with
         * variance of added noise.  
         * @param[in,out] data The Image to be noise-ified.
         * @param[in,out] variance Image to fill with variance of applied noise.  Will be resized
         * if it does not match dimensions of data.
         */
        void operator() (Image<> data, Image<> variance) {
        }
    };
};  // namespace galsim

#endif
