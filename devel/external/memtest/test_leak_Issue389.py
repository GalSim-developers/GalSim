"""@file test_leak_Issue389.py

This is short module designed to test a memory leak that Barney (barnaby.t.p.rowe@gmail.com) was
finding with tests on GalSim Issue #389.  This failure seems to be localized to the use of
fink-provided gcc, specifically gcc 4.6.4, 4.7.3, and 4.8.1.  Using these compilers, the `for` loop
at the end of this script causes memory to be eaten up rapidly on a MacBook Pro running OSX 10.7.5.

This issue also only seems to affect the use of `k_interpolant=galsim.Lanczos(n)`.

Interestingly, for these fink gcc compilers the `scons` option `MEM_TEST=True` also produces errors
that indicate some problem in the installation:
```
/sw/bin/g++-fsf-4.6 -o src/.obj/BinomFact.os -c -O2 -fno-strict-aliasing -Wall -Werror -fPIC -DMEM_TEST -Iinclude/galsim -Iinclude -I/Users/browe/local/include -I/sw/include src/BinomFact.cpp

In file included from /sw/lib/gcc4.6/lib/gcc/x86_64-apple-darwin11.4.2/4.6.4/../../../../include/c++/4.6.4/vector:63:0,
                 from src/BinomFact.cpp:24:
/sw/lib/gcc4.6/lib/gcc/x86_64-apple-darwin11.4.2/4.6.4/../../../../include/c++/4.6.4/bits/stl_construct.h: In function 'void std::_Construct(_T1*, const _T2&)':
/sw/lib/gcc4.6/lib/gcc/x86_64-apple-darwin11.4.2/4.6.4/../../../../include/c++/4.6.4/bits/stl_construct.h:84:9: error: expected id-expression before '(' token
scons: *** [src/.obj/BinomFact.os] Error 1
scons: building terminated because of errors.
```

So far, the workaround is to move to using the system `/usr/bin/g++` (from gcc 4.2), which works
without problems.  But this is worth further investigation and maybe a bug report to fink or gcc.
"""

try:
    import galsim
except ImportError:
    import os
    import sys
    path, filename = os.path.split(__file__)
    sys.path.append(os.path.abspath(os.path.join(path, "..", "..", "..")))
    import galsim

sersic = galsim.Sersic(n=3.1, half_light_radius=0.6)

im = galsim.ImageD(512, 512)

sersic.draw(im, dx=0.03)

interpolated = galsim.InterpolatedImage(im, dx=0.03, k_interpolant=galsim.Lanczos(3))
interpolated_convolved = galsim.Convolve([interpolated, galsim.Gaussian(1.e-8)])

outimage = galsim.ImageD(512, 512)
for i in range(8):
    interpolated_convolved.draw(outimage, dx=0.03)
