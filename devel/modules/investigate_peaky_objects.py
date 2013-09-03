import galsim


image_size = 512 
pixel_scale = 0.03

def make_images():
    # Make some images to write to
    test_image = galsim.ImageD(image_size,image_size)
    test_image.setScale(pixel_scale)
    test_image_interp = galsim.ImageD(image_size,image_size)
    test_image_interp.setScale(pixel_scale)
    test_image_param = galsim.ImageD(image_size,image_size)
    test_image_param.setScale(pixel_scale)
    sersic_image = galsim.ImageD(image_size, image_size)
    sersic_image.setScale(pixel_scale)
    # Create two objects to add
    galaxy1 = galsim.Sersic(n=4., half_light_radius = 0.5)
    galaxy2 = galsim.Gaussian(sigma=1.*pixel_scale)
    overall = 0.1*galaxy1+galaxy2 # Make the Sersic fainter...
    overall.draw(test_image, dx=pixel_scale)
    # Generate an InterpolatedImage of that thing
    overall_interp = galsim.InterpolatedImage(test_image,dx=pixel_scale)
    overall_interp.applyShear(g1=0.05, g2=0.) # shear it
    overall_interp.draw(test_image_interp,dx=pixel_scale) # draw it
    test_image_interp.write('sersic-interp.fits')
    galaxy1.applyShear(galsim.Shear(g1=0.05,g2=0.)) # then apply the same shear to 
    galaxy2.applyShear(galsim.Shear(g1=0.05,g2=0.)) # the analytic profiles
    overall = 0.1*galaxy1+galaxy2 # add them
    overall.draw(test_image_param, dx=pixel_scale) # draw it
    test_image_param.write('sersic-param.fits')
    test_image = test_image_interp-test_image_param # make a difference image 
    test_image.write('sersic-diff.fits')

if __name__ == "__main__":

    make_images()

