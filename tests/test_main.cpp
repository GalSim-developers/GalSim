#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Main

// The boost unit tests have some unused variables, so suppress the warnings about that.
#if defined(__GNUC__) && __GNUC__ >= 4 && (__GNUC__ >= 5 || __GNUC_MINOR__ >= 2)
// I think pragma GCC was introduced in gcc 4.2
#pragma GCC diagnostic ignored "-Wunused-variable"
#endif

#include <boost/test/included/unit_test.hpp>

//JAZ Nothing needs to go here - the test module definitions above create a main function.
