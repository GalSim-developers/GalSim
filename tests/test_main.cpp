#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Main

// icpc pretends to be GNUC, since it thinks it's compliant, but it's not.
// It doesn't understand "pragma GCC"
#ifndef __INTEL_COMPILER

// The boost unit tests have some unused variables, so suppress the warnings about that.
// I think pragma GCC was introduced in gcc 4.2, so guard for >= that version 
#if defined(__GNUC__) && __GNUC__ >= 4 && (__GNUC__ >= 5 || __GNUC_MINOR__ >= 2)
#pragma GCC diagnostic ignored "-Wunused-variable"
#endif

// Not sure when this was added.  Currently check for it for versions >= 4.3
#if defined(__GNUC__) && __GNUC__ >= 4 && (__GNUC__ >= 5 || __GNUC_MINOR__ >= 3)
#pragma GCC diagnostic ignored "-Warray-bounds"
#endif

// Only clang seems to have this
#ifdef __clang__
#if __has_warning("-Wlogical-op-parentheses")
#pragma GCC diagnostic ignored "-Wlogical-op-parentheses"
#endif
#endif

#endif // !INTEL

#include <boost/test/included/unit_test.hpp>

//JAZ Nothing needs to go here - the test module definitions above create a main function.
