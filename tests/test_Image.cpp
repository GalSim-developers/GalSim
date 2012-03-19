#include "galsim/Image.h"
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>



BOOST_AUTO_TEST_SUITE(image_tests);

const int nrow=100;
const int ncol=150;

BOOST_AUTO_TEST_CASE( image_pixel_range_definition )
{
	int nrow=100;
	int ncol=150;
	galsim::Image<int> img(nrow,ncol);
    BOOST_CHECK(img.XMin()==1);
    BOOST_CHECK(img.XMax()==nrow);
    BOOST_CHECK(img.YMin()==1);
    BOOST_CHECK(img.YMax()==ncol);
}

BOOST_AUTO_TEST_CASE( image_creation )
{
	
}

BOOST_AUTO_TEST_CASE( image_data )
{
	int nrow=100;
	int ncol=150;
	galsim::Image<int> img(nrow,ncol);
	BOOST_CHECK(img.data() != NULL);
}


BOOST_AUTO_TEST_SUITE_END();
