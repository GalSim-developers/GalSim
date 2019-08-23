import numpy as np
import galsim
import time
import cProfile
import pstats

import argparse

ntrial = 100


class Sim(object):
    def __init__(self, shear=False):
        self.shear=shear
        self.dim=100

        self.flux = 100.0
        self.hlr = 4.0
        self.n_knots=100
        self.psf=galsim.Gaussian(fwhm=0.9)

    def make_im(self):
        obj=self.get_model()
        if self.shear:
            g1,g2=self.get_shape()
            obj = obj.shear(g1=g1,g2=g2)

        obj=galsim.Convolve(obj, self.psf)
        return obj.drawImage(
            scale=0.263,
        ).array

    def get_shape(self):
        while True:
            g1,g2 = np.random.normal(scale=0.2,size=2)
            g=np.sqrt(g1**2 + g2**2)
            if g < 0.99:
                break
        return g1,g2

    def get_model(self):
        profile=galsim.Exponential(
            flux=self.flux,
            half_light_radius=self.hlr,
        )

        obj=galsim.RandomWalk(
            self.n_knots,
            profile=profile,
        )
        return obj

for shear in [False, True]:

    sim=Sim(shear=shear)

    pr = cProfile.Profile()
    pr.enable()

    t0 = time.time()
    for i in range(ntrial):
        im=sim.make_im()
    t1 = time.time()

    pr.disable()
    ps = pstats.Stats(pr).sort_stats('time')
    #ps.print_stats(20)

    print("time for shear=%s:"%shear, t1-t0)
