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
    print(f"{j:>2d} {len(Z.coef)} {t1-t0:6.3f} s")


def g(j):
    rng = galsim.BaseDeviate(191121).as_numpy_generator()
    t0 = time.time()
    xy_outer = 8.36/2
    xy_inner = xy_outer*0.612
    uv_outer = 1.75
    uv_inner = 0.0
    coef = rng.uniform(-0.1, 0.1, size=(j+1, j+1))
    coef[0] = 0.0
    coef[:, 0] = 0.0
    dz = galsim.zernike.DoubleZernike(
        coef,
        xy_inner=xy_inner, xy_outer=xy_outer,
        uv_inner=uv_inner, uv_outer=uv_outer
    )
    t0 = time.time()
    dz._coef_array_xyuv
    del dz.coef
    dz.coef
    t1 = time.time()
    print(f"{j:>2d} {t1-t0:6.3f} s")


if __name__ == "__main__":
    for j in [6, 10, 15, 21, 28, 36]:
        f(j)
        # g(j)
