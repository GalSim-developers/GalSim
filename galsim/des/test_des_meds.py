import sys
sys.path.append('/home/tomek/Work/code/GalSim/galsim/des/')
import des_meds
import numpy


# get two objects, each with two exposures

objlist = []

# first obj

img11 = numpy.ones([32,32])*111
img12 = numpy.ones([32,32])*112
seg11 = numpy.ones([32,32])*121
seg12 = numpy.ones([32,32])*122
wth11 = numpy.ones([32,32])*131
wth12 = numpy.ones([32,32])*132
jac11 = numpy.array([[11.1 , 11.2],[12.3 , 11.4]])
jac12 = numpy.array([[12.1 , 12.2],[12.3 , 12.4]])

images =  [img11,img12]
weights = [wth11,wth12]
segs =    [seg11,seg12]
jacs = 	  [jac11,jac12]

obj1 = des_meds.MultiExposureObject(images,weights,segs,jacs)

img21 = numpy.ones([32,32])*211
img22 = numpy.ones([32,32])*212
seg21 = numpy.ones([32,32])*221
seg22 = numpy.ones([32,32])*222
wth21 = numpy.ones([32,32])*231
wth22 = numpy.ones([32,32])*232
jac21 = numpy.array([[21.1 , 21.2],[22.3 , 21.4]])
jac22 = numpy.array([[22.1 , 22.2],[22.3 , 22.4]])

images =  [img21,img22]
weights = [wth21,wth22]
segs =    [seg21,seg22]
jacs = 	  [jac22,jac22]

obj2 = des_meds.MultiExposureObject(images,weights,segs,jacs)

objlist = [obj1,obj2]

filename_meds='test_meds.fits'
des_meds.write(filename_meds,objlist,clobber=True)
print 'wrote meds file %s ' % filename_meds

# test functions in meds.py

print 'reading %s' % filename_meds
m=des_meds.MEDS(filename_meds)
print 'number of objects is %d' % len(m._cat)


print 'testing if loaded images are the same as original images'

cat=m.get_cat()

n_obj=2
n_cut=2
for iobj in range(n_obj):
	for icut in range(n_cut):

		img=m.get_cutout( iobj, icut, type='image')
		wth=m.get_cutout( iobj, icut, type='weight')
		seg=m.get_cutout( iobj, icut, type='seg')
		jac=numpy.zeros([2,2])
		jac[0,0]=cat['dudrow'][iobj,icut]
		jac[0,1]=cat['dudcol'][iobj,icut]
		jac[1,0]=cat['dvdrow'][iobj,icut]
		jac[1,1]=cat['dvdcol'][iobj,icut]

		# import pdb;pdb.set_trace()

		numpy.testing.assert_array_equal(img,objlist[iobj].images[icut])	
		numpy.testing.assert_array_equal(wth,objlist[iobj].weights[icut])	
		numpy.testing.assert_array_equal(seg,objlist[iobj].segs[icut])	
		numpy.testing.assert_array_equal(jac,objlist[iobj].jacs[icut])	

		print 'passed obj=%d icut=%d' % (iobj,icut)

print 'all asserts succeeded'