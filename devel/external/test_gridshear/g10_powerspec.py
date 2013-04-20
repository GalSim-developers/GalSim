"""
Power spectrum calculation for great10 variable shear catalogue, in python/numpy.

Original code in matlab from T. Kitching (April 2011) : If you use this code, or any derivative
of this code, in any publication, you must reference Kitching et al. (2010) (GREAT10 Handbook)
and acknowledge Thomas Kitching. 

Ported to python by Malte Tewes and Guldariya Nurbaeva (EPFL, May 2011).
Consider this as an "example", as work in progress, use at your own risk !
(Having said this, the code below seems to give identical results than the matlab version.)

A demo usage is provided at the bottom of this file.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys

class ps:
	"""
	A class to calculate the power spectrum.
	"""
	def __init__(self, g, step = 48.0, size = 10.0, nbin2 = 20):
		"""
		Attributes :
		- "g" is a complex square array representing a measurement at each point of a square grid.
		- "step" is the pixel step size in both x and y = postage stamp size = 48 for great10
		- "size" is is the total grid width (= height), in degrees, 10 for great10	
		- "nbin2" : number of ell bins for the power spectrum
		"""
		self.g = g
		self.step = step
		self.size = size
		self.nbin2 = nbin2
		
	
	def setup(self):
		"""
		Set up stuff like l ranges
		"""
		self.n = self.g.shape[0] # The width = height of the array
		if self.g.shape[1] != self.n:
			sys.exit("Only square arrays !")
	
		self.radstep =  (self.size / self.n) * (np.pi / 180.0) # Angular step size in radians
		
		bigl = self.size * np.pi / 180.0
		
		self.max_l_mode = 2.0 * np.pi / self.radstep
		self.min_l_mode = 2.0 * np.pi / (self.size * np.pi/180.0)
		self.nyquist_deg = self.size / self.n
		
		print "Range of l modes : %f to %f" % (self.min_l_mode, self.max_l_mode)
		
		#print "Builing logarithmics l bins ..."
		self.dlogl = (self.max_l_mode - self.min_l_mode)/(self.nbin2 - 1.0)
		lbin = self.min_l_mode + (self.dlogl * (np.arange(self.nbin2))) -1.0 + 0.00001
                #print lbin
                #print self.dlogl
		
		nbin = 2 * self.n

		# Creating a complex wavevector
		
		self.el1 = 2.0 * np.pi * (np.arange(self.n)  - ((self.n-1)/2.0) - 0.5 + 0.001) / bigl

		self.lvec = np.zeros((self.n,self.n), dtype = np.complex)
		icoord = np.zeros((self.n,self.n))
		jcoord = np.zeros((self.n,self.n))
		
		for i1 in range(self.n): # warning different python/matlab convention, i1 starts at 0
			l1 = self.el1[i1]
			for j1 in range(self.n):
				l2 = self.el1[j1]     
				self.lvec[i1,j1] = np.complex(l1, l2)
				icoord[i1,j1] = i1+1
				jcoord[i1,j1] = j1+1
		
		
		
	def create(self):
		"""
		Calculate the actual power spectrum
		"""
		#% Estimate E and B modes assuming linear-KS.
		gfieldft = np.fft.fftshift(np.fft.fft2(self.g))
		gkapi = np.conjugate(self.lvec) * np.conjugate(self.lvec) * gfieldft / (self.lvec * np.conjugate(self.lvec))
		gkapi = np.fft.ifft2(np.fft.ifftshift(gkapi))
		
		gkapft = np.fft.fftshift(np.fft.fft2(np.real(gkapi)))
		gbetft = np.fft.fftshift(np.fft.fft2(np.imag(gkapi)))
		
		self.gCEE_2 = np.real(gkapft)**2.0 + np.imag(gkapft)**2.0 # E mode power
		self.gCBB_2 = np.real(gbetft)**2.0 + np.imag(gbetft)**2.0 # B mode power
		self.gCEB_2 = np.dot(np.real(gkapft), np.real(gbetft)) - np.dot(np.imag(gkapft), np.imag(gbetft)) # EB cross power
		
		
		
	def angavg(self):
		"""
		Angular average of the spectrum
		"""
		
		self.gPowEE = np.zeros(self.nbin2)
		self.gPowBB = np.zeros(self.nbin2)
		self.gPowEB = np.zeros(self.nbin2)
		self.ll = np.zeros(self.nbin2)
		dll = np.zeros(self.nbin2)
		
		for i1 in range(self.n): # start at 0
			l1 = self.el1[i1]
			for j1 in range(self.n):
				l2 = self.el1[j1]
				l = np.sqrt(l1*l1 + l2*l2)
				#print l
				
				if ( l <= self.max_l_mode and l >= self.min_l_mode) :
					ibin = int(np.round((l + 1 - self.min_l_mode) / self.dlogl))
					self.gPowEE[ibin] += self.gCEE_2[i1,j1] * l
					self.gPowBB[ibin] += self.gCBB_2[i1,j1] * l
					self.gPowEB[ibin] += self.gCEB_2[i1,j1] * l
				else:
					print "Hmm, l out of min-max range, this part should be improved ..."
					
				self.ll[ibin] = l # the array of l values
				if ibin > 1:
            				dll[ibin] = self.ll[ibin+1] - self.ll[ibin] # ibin starts from 0
		
		self.gPowEE /= (self.n**4 * self.dlogl)
		self.gPowBB /= (self.n**4 * self.dlogl)
		self.gPowEB /= (self.n**4 * self.dlogl)

	
	def plot(self, title="Power Spectrum"):
		"""
		Plot it
		"""
		plt.loglog(self.ll, self.gPowEE, "r.-", label="E mode")
		plt.loglog(self.ll, self.gPowBB, "b.-", label="B mode")
		plt.xlabel("Wavenumber l")
		plt.ylabel("Power [l^2 C_l / (2 pi)]")
		plt.title(title)
		plt.legend()
		plt.show()
		

def readells(filepath, n = 100, step = 48.0):
	"""
	Read a great10 dat file an turn it into a complex numpy array of g or e. 
	The dat files contains 4 columns :
	e1 e2 x y (or g1 g2 x y)
	where x and y are in *pixels*
	We turn this into an array with indexes, in a slow but somewhat pseudo-safe way...
	- n is the height and width of the square n x n array, 100 for great10
	- step = step between measurements in pixels, 48 for great10
	"""

	data = np.loadtxt(filepath).transpose()
	print "Reading %s ..." % (filepath)
	print "Input data shape : %s" % (str(data.shape))

	e1 = data[0]
	e2 = -data[1]
	x = data[2]
	y = data[3]

	# The complex array in which we will store the e :
	ein = np.zeros((n,n), dtype = np.complex)
	
	for i in range(n * n):
		# warning : python convention, first index is 0
		jindex = int(x[i]/step + 0.5) - 1
		iindex = int(y[i]/step + 0.5) - 1
		if iindex>=0 and iindex <len(x) and jindex>=0 and jindex<len(x) :
			ein[iindex,jindex] = np.complex(e1[i], e2[i])
		else:
			print "OUCH, index out of bounds."
			sys.exit()

	return ein
	

if __name__ == "__main__":
	
	ein = readells("g10_v1_training_1.truth.dat")
	myps = ps(ein)
	myps.setup()
	myps.create()
	myps.angavg()
	myps.plot(title = "Training 1 true shear")
	

