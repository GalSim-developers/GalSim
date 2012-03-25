import galsim
import numpy as np

testshape = (512, 512)  # shape of image arrays for all tests
types = [np.int16, np.int32, np.float32, np.float64]

def test_Image_XYmin_XYMax():
    """Test that all four types of supported arrays correctly set bounds based on an input array.
    """
    for array_type in types:
        test_array = np.zeros(testshape, dtype=array_type)
        image = galsim.Image[array_type](test_array)
        assert image.getXMin() == 1
        assert image.getYMin() == 1
        assert image.getXMax() == testshape[0]
        assert image.getYMax() == testshape[1]

def test_Image_array_view():
    """Test that all four types of supported arrays correctly provide a view on an input array.
    """
    for array_type in types:
        test_array = (np.random.random_integers(low=0, high=1, size=testshape)).astype(array_type)
        image = galsim.Image[array_type](test_array)
        np.testing.assert_array_equal(test_array, image.array,
                                      err_msg="Array look into Image class does not match input.")
 


