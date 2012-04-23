#include "galsim/Image.h"
#define BOOST_TEST_DYN_LINK

// The boost unit tests have some unused variables, so suppress the warnings about that.
#ifdef __GNUC__
#pragma GCC diagnostic ignored "-Wunused-variable"
#endif
#include <boost/test/unit_test.hpp>


BOOST_AUTO_TEST_SUITE(image_tests);

const int nrow=100;
const int ncol=150;

BOOST_AUTO_TEST_CASE( image_pixel_range_definition )
{
	int nrow=100;
	int ncol=150;
	galsim::Image<int> img(nrow,ncol);
    BOOST_CHECK(img.getXMin()==1);
    BOOST_CHECK(img.getXMax()==nrow);
    BOOST_CHECK(img.getYMin()==1);
    BOOST_CHECK(img.getYMax()==ncol);
}

BOOST_AUTO_TEST_CASE( image_creation )
{
	
}

BOOST_AUTO_TEST_CASE( image_data )
{
	int nrow=100;
	int ncol=150;
	galsim::Image<int> img(nrow,ncol);
	BOOST_CHECK(img.getData() != NULL);
}


BOOST_AUTO_TEST_SUITE_END();
