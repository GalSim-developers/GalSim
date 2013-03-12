import galsim
import pylab
import numpy as np
import time



def scale_free_spectrum(k):
	p = np.zeros(k.shape)
	w = (k!=0)
	p[w]=k[w]**-2
	return p


def show_shear(E,B,title):
	R=galsim.lensing.PowerSpectrumRealizer(160, 160, 0.1, E, B)
	gd=galsim.GaussianDeviate(lseed=1234567)

	g1, g2, kappa = R(gd, get_kappa=True)

	g = (g1**2 + g2**2) ** 0.5
	theta = 0.5*np.arctan2(-g2, g1)
	gx = g * np.cos(theta)
	gy = g * np.sin(theta)

	pylab.figure()
	pylab.pcolor(kappa)
	pylab.colorbar()
	pylab.quiver(gx, gy, scale=50, headwidth=0, pivot='middle')
	pylab.title(title)
        pylab.figure()
        ke, kb = galsim.lensing.invert_kappa_from_shear(g1, -g2)
        pylab.pcolor(ke); pylab.colorbar()
        pylab.title('kappa_E')
        pylab.quiver(gx, gy, scale=50, headwidth=0, pivot='middle')
        pylab.figure()
        pylab.pcolor(kb); pylab.colorbar()
        pylab.title('kappa_B')

show_shear(scale_free_spectrum, None, "E-mode")
#show_shear(None, scale_free_spectrum, "B-mode")


pylab.show()
