# A simple exponential fitter that we can use to time the k-space drawing function

from __future__ import print_function
import galsim
import scipy.optimize
import numpy

import time
import cProfile, pstats
pr = cProfile.Profile()

def draw(params, image=None):
    """Draw an Exponential profile in k-space onto the given image

    params = [ half_light_radius, flux, e1, e2 ]
    image is optional, but if provided will be used as is.
    """
    exp = galsim.Exponential(params[0], flux=params[1]).shear(g1=params[2], g2=params[3])
    if image is None:   
        image = exp.drawKImage()
    else:
        #image = exp._drawKImage(image)
        image = exp.drawKImage(image)
    return image

def fit(image, guess=(1.,1.,0.,0.), tol=1.e-6):
    """Find the best fitting exponential to the given k-space image
    """

    class resid(object):
        def __init__(self, image):
            self._target_image = image.copy()
            self._scratch_image = image.copy()
        def __call__(self, params):
            if params[0] < 0. or abs(params[2]) > 1. or abs(params[3]) > 1.:
                return 1.e500
            try:
                draw(params, self._scratch_image)
            except ValueError:
                return 1.e500
            a = self._scratch_image.array
            a -= self._target_image.array
            a **= 2
            return numpy.abs(numpy.sum(a))

    guess = numpy.array(guess)
    print('guess = ',guess)
    result = scipy.optimize.minimize(resid(image), guess, tol=tol)
    print('result = ',result.x)
    print('number of iterations = ',result.nit)
    print('number of function evals = ',result.nfev)
    return result.x

true_params = [3.49, 99.123, 0.0812, -0.2345]
true_image = draw(true_params)

pr.enable()
t0 = time.time()
fit_params = fit(true_image, tol=1.e-12)
t1 = time.time()
pr.disable()

ps = pstats.Stats(pr).sort_stats('time')
ps = pstats.Stats(pr).sort_stats('cumtime')
ps.print_stats(20)

print('True params = ',true_params)
print('Fitted params = ',fit_params.tolist())
print('time = ',t1-t0)
