import galsim
import galsim.lsst
import time
import os

gsp = galsim.GSParams(maximum_fft_size=8192)

print 'No spider...'
t1 = time.time()
psf1 = galsim.lsst.getPSF(no_spider=True, gsparams=gsp)
im1 = psf1.drawImage(scale=0.02)
print 'Done: %f sec'%(time.time()-t1)
print ''

print 'Approximate spider...'
t1 = time.time()
psf2 = galsim.lsst.getPSF(approximate_spider=True, gsparams=gsp)
print 'step and max K:',psf2.stepK(), psf2.maxK()
im2 = psf2.drawImage(scale=0.02)
print 'Done: %f sec'%(time.time()-t1)
print ''

#psf2 = galsim.lsst.getPSF(approximate_spider=True, gsparams=gsp, strut_thick=0.02)
#im2_2 = psf2.drawImage(scale=0.02)

#psf2 = galsim.lsst.getPSF(approximate_spider=True, gsparams=gsp, strut_thick=0.06)
#im2_3 = psf2.drawImage(scale=0.02)

#psf2 = galsim.lsst.getPSF(approximate_spider=True, gsparams=gsp, strut_thick=0.14)
#im2_4 = psf2.drawImage(scale=0.02)

if True:
    print 'Real spider...'
    t1 = time.time()
    psf3 = galsim.lsst.getPSF(gsparams=gsp)
    print 'Initialized: %f sec'%(time.time()-t1)
    print 'step and max K:',psf3.stepK(), psf3.maxK()
    im3 = psf3.drawImage(scale=0.02)
    print 'Total time: %f sec'%(time.time()-t1)
    print ''

if False:
    print 'Real spider...'
    t1 = time.time()
    im = galsim.Image(galsim.fits.read(os.path.join(galsim.meta_data.share_dir, "lsst_spider_2048.fits.gz")))
    print 'Loaded image separately: %f sec'%(time.time()-t1)
    psf4 = galsim.lsst.getPSF(gsparams=gsp, pupil_plane_im=im)
    print 'Initialized after %f sec'%(time.time()-t1)
    im4 = psf4.drawImage(scale=0.02)
    print 'Total time: %f sec'%(time.time()-t1)
    print ''

if True:
    foo = galsim.Kolmogorov(fwhm=0.65)
    print 'Atmospheric PSF step and max K:',foo.stepK(), foo.maxK()

    t1 = time.time()
    bar = galsim.Convolve(psf3,foo)
    im5 = bar.drawImage(scale=0.2)
    print 'time to draw real spider PSF convolved with Kolmogorov:',time.time()-t1
    print 'it has step and max K:',bar.stepK(), bar.maxK()
    print 'it has goodImageSize:',bar.SBProfile.getGoodImageSize(0.2,1.)
    print ''

    print 'Real spider, 512x512'
    t1 = time.time()
    psf10 = galsim.lsst.getPSF(gsparams=gsp, im_size=512)
    print 'Initialized: %f sec'%(time.time()-t1)
    print 'step and max K:',psf10.stepK(), psf10.maxK()
    im10 = psf10.drawImage(scale=0.02)
    print 'Total time: %f sec'%(time.time()-t1)

    t1 = time.time()
    bar = galsim.Convolve(psf10,foo)
    print 'time to convolve with Kolmogorov:',time.time()-t1
    t1 = time.time()
    im10_2 = bar.drawImage(scale=0.2)
    print 'time to draw convolved with K:',time.time()-t1
    print ''

if False:
    print 'Real spider, 256x256'
    t1 = time.time()
    psf11 = galsim.lsst.getPSF(gsparams=gsp, im_size=256)
    print 'Initialized: %f sec'%(time.time()-t1)
    print 'step and max K:',psf11.stepK(), psf11.maxK()
    im11 = psf11.drawImage(scale=0.02)
    print 'Total time: %f sec'%(time.time()-t1)

if False:
    t1 = time.time()
    bar = galsim.Convolve(psf11,foo)
    print 'time to convolve with Kolmogorov:',time.time()-t1
    t1 = time.time()
    im11_2 = bar.drawImage(scale=0.2)
    print 'time to draw convolved with K:',time.time()-t1
    print ''

if False:
    print 'Real spider, 256x256, change oversampling'
    #gsp = galsim.GSParams(maximum_fft_size=8192,folding_threshold=2.e-3)
    t1 = time.time()
    psf21 = galsim.lsst.getPSF(gsparams=gsp, im_size=256, oversampling=2)
    print 'Initialized: %f sec'%(time.time()-t1)
    print 'step and max K:',psf21.stepK(), psf21.maxK()
    im21 = psf21.drawImage(scale=0.02)
    print 'Total time: %f sec'%(time.time()-t1)

if False:
    t1 = time.time()
    bar = galsim.Convolve(psf21,foo)
    print 'time to convolve with Kolmogorov:',time.time()-t1
    t1 = time.time()
    im21_2 = bar.drawImage(scale=0.2)
    print 'time to draw convolved with K:',time.time()-t1
    print ''

    #print 'Real spider oversampling=2...'
#gsp = galsim.GSParams(maximum_fft_size=16384)
    #t1 = time.time()
    #psf4 = galsim.lsst.getPSF(gsparams=gsp, oversampling=2)
    #im4 = psf4.drawImage(scale=0.02)
    #print 'Done: %f sec'%(time.time()-t1)

    #print 'Real spider pad_factor=2...'
#gsp = galsim.GSParams(maximum_fft_size=16384)
    #t1 = time.time()
    #psf5 = galsim.lsst.getPSF(gsparams=gsp, pad_factor=2)
    #im5 = psf5.drawImage(scale=0.02)
    #print 'Done: %f sec'%(time.time()-t1)

    #print 'Real spider highres...'
#gsp = galsim.GSParams(maximum_fft_size=16384)
    #t1 = time.time()
    #psf6 = galsim.lsst.getPSF(gsparams=gsp, highres=True)
    #im6 = psf6.drawImage(scale=0.02)
    #print 'Done: %f sec'%(time.time()-t1)


im1.write('no_spider.fits')
im2.write('approx_spider_st0.1.fits')
#im2_2.write('approx_spider_st0.02.fits')
#im2_3.write('approx_spider_st0.06.fits')
#im2_4.write('approx_spider_st0.14.fits')
if True:
    im3.write('real_spider.fits')
    im5.write('real_spider_kolm.fits')
    im10.write('real_spider_512.fits')
    im10_2.write('real_spider_512_kolm.fits')

if False:
    #im4.write('real_spider_o2.fits')
    #im5.write('real_spider_p2.fits')
    #im6.write('real_spider_highres.fits')
    #im6.write('real_spider_force_maxk_kolm.fits')
    im11.write('real_spider_256.fits')
    im11_2.write('real_spider_256_kolm.fits')
    im21.write('real_spider_256f.fits')
    im21_2.write('real_spider_256f_kolm.fits')


