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


obj={}
obj['image'] =  [img11,img12]
obj['weight'] = [wth11,wth12]
obj['seg'] =    [seg11,seg12]
obj['dudrow'] = 111
obj['dudcol'] = 112
obj['dvdrow'] = 121
obj['dvdcol'] = 122

objlist.append(obj)

img21 = numpy.ones([32,32])*211
img22 = numpy.ones([32,32])*212
seg21 = numpy.ones([32,32])*221
seg22 = numpy.ones([32,32])*222
wth21 = numpy.ones([32,32])*231
wth22 = numpy.ones([32,32])*232


obj={}
obj['image'] =  [img21,img22]
obj['weight'] = [wth21,wth22]
obj['seg'] =   [seg21,seg22]
obj['dudrow'] = 211
obj['dudcol'] = 212
obj['dvdrow'] = 221
obj['dvdcol'] = 222

objlist.append(obj)

filename_meds='test_meds.fits'
des_meds.write(filename_meds,objlist,clobber=True)
print 'wrote meds file %s ' % filename_meds

# test functions in meds.py

print 'reading %s' % filename_meds
m=des_meds.MEDS(filename_meds)
print 'number of objects is %d' % len(m._cat)

print 'testing if loaded images are the same as original images'
# check first object

iobj=0
img1=m.get_cutout( iobj, 0, type='image')
img2=m.get_cutout( iobj, 1, type='image')
seg1=m.get_cutout( iobj, 0, type='seg')
seg2=m.get_cutout( iobj, 1, type='seg')
wth1=m.get_cutout( iobj, 0, type='weight')
wth2=m.get_cutout( iobj, 1, type='weight')

numpy.testing.assert_array_equal(img11,objlist[iobj]['image'][0])	
numpy.testing.assert_array_equal(img12,objlist[iobj]['image'][1])	
numpy.testing.assert_array_equal(seg11,objlist[iobj]['seg'][0])	
numpy.testing.assert_array_equal(seg12,objlist[iobj]['seg'][1])
numpy.testing.assert_array_equal(wth11,objlist[iobj]['weight'][0])	
numpy.testing.assert_array_equal(wth12,objlist[iobj]['weight'][1])	


iobj=1
img1=m.get_cutout( iobj, 0, type='image')
img2=m.get_cutout( iobj, 1, type='image')
seg1=m.get_cutout( iobj, 0, type='seg')
seg2=m.get_cutout( iobj, 1, type='seg')
wth1=m.get_cutout( iobj, 0, type='weight')
wth2=m.get_cutout( iobj, 1, type='weight')

numpy.testing.assert_array_equal(img21,objlist[iobj]['image'][0])	
numpy.testing.assert_array_equal(img22,objlist[iobj]['image'][1])	
numpy.testing.assert_array_equal(seg21,objlist[iobj]['seg'][0])	
numpy.testing.assert_array_equal(seg22,objlist[iobj]['seg'][1])
numpy.testing.assert_array_equal(wth21,objlist[iobj]['weight'][0])	
numpy.testing.assert_array_equal(wth22,objlist[iobj]['weight'][1])	


print 'all asserts succeeded'