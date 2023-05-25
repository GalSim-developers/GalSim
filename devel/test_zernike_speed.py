import numpy as np
import galsim
import time


def f(j):
    bd = galsim.BaseDeviate(191120)
    u = galsim.UniformDeviate(bd)
    t0 = time.time()
    for _ in range(100):
        R_inner = u()*0.5+0.2
        R_outer = u()*2.0+2.0
        # Create Zernike.  Populates .coef
        Z1 = galsim.zernike.Zernike([0]+[u() for i in range(j)], R_outer=R_outer, R_inner=R_inner)
        Z2 = galsim.zernike.Zernike([0]+[u() for i in range(j)], R_outer=R_outer, R_inner=R_inner)
        Z = Z1*Z2
        Z.coef
    t1 = time.time()
    print(f"{j:>2d} {t1-t0:6.3f} s")


if __name__ == "__main__":
    for j in [11, 22, 36]:
        f(j)
