// -*- c++ -*-
/*
 * Copyright 2012, 2013 The GalSim developers:
 * https://github.com/GalSim-developers
 *
 * This file is part of GalSim: The modular galaxy image simulation toolkit.
 *
 * GalSim is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * GalSim is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with GalSim.  If not, see <http://www.gnu.org/licenses/>
 */
#include "galsim/Image.h"
#define BOOST_TEST_DYN_LINK

// icpc pretends to be GNUC, since it thinks it's compliant, but it's not.
// It doesn't understand #pragma GCC
// Rather, it uses #pragma warning(disable:nn)
#ifdef __INTEL_COMPILER

// Disable "overloaded virtual function ... is only partially overridden"
#pragma warning(disable:654)

#else

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

#endif

#include <boost/test/unit_test.hpp>
#include <boost/test/test_case_template.hpp>
#include <boost/mpl/list.hpp>

BOOST_AUTO_TEST_SUITE(image_tests);

typedef boost::mpl::list<short, int, float, double> test_types;

BOOST_AUTO_TEST_CASE_TEMPLATE( TestImageBasic , T , test_types )
{
    // Tests are designed for these to be odd, but not necessarily equal
    const int ncol=7;  // x ranges from 1 .. ncol
    const int nrow=5;  // y ranges from 1 .. nrow
    T ref_array[nrow*ncol] = {
        // x  -> 
        11, 21, 31, 41, 51, 61, 71,  // y
        12, 22, 32, 42, 52, 62, 72,  //  
        13, 23, 33, 43, 53, 63, 73,  // |
        14, 24, 34, 44, 54, 64, 74,  // v
        15, 25, 35, 45, 55, 65, 75 };
    // Of course, when viewed as an image, the rows are generally drawn from bottom to top.

    // Check basic constructor from nrow,ncol
    galsim::Image<T> im1(ncol,nrow);
    galsim::Bounds<int> bounds(1,ncol,1,nrow);

    BOOST_CHECK(im1.getXMin()==1);
    BOOST_CHECK(im1.getXMax()==ncol);
    BOOST_CHECK(im1.getYMin()==1);
    BOOST_CHECK(im1.getYMax()==nrow);
    BOOST_CHECK(im1.getBounds()==bounds);

    BOOST_CHECK(im1.getData() != NULL);
    BOOST_CHECK(im1.getStride() == ncol);

    // Check alternate constructor from bounds
    galsim::Image<T> im2(bounds);
    galsim::ImageView<T> im2_view = im2;
    galsim::ConstImageView<T> im2_cview = im2;

    BOOST_CHECK(im2_view.getXMin()==1);
    BOOST_CHECK(im2_view.getXMax()==ncol);
    BOOST_CHECK(im2_view.getYMin()==1);
    BOOST_CHECK(im2_view.getYMax()==nrow);
    BOOST_CHECK(im2_view.getBounds()==bounds);

    BOOST_CHECK(im2_cview.getXMin()==1);
    BOOST_CHECK(im2_cview.getXMax()==ncol);
    BOOST_CHECK(im2_cview.getYMin()==1);
    BOOST_CHECK(im2_cview.getYMax()==nrow);
    BOOST_CHECK(im2_cview.getBounds()==bounds);

    BOOST_CHECK(im2.getData() != NULL);
    BOOST_CHECK(im2_view.getData() == im2.getData());
    BOOST_CHECK(im2_cview.getData() == im2.getData());
    BOOST_CHECK(im2.getStride() == ncol);
    BOOST_CHECK(im2_view.getStride() == ncol);
    BOOST_CHECK(im2_cview.getStride() == ncol);

    // Check various ways to set and get values 
    for (int y=1; y<=nrow; ++y) {
        for (int x=1; x<=ncol; ++x) {
            im1(x,y) = 100 + 10*x + y;
            im2_view(x,y) = 100 + 10*x + y;
        }
    }
    for (int y=1; y<=nrow; ++y) {
        for (int x=1; x<=ncol; ++x) {
            BOOST_CHECK(im1.at(x,y) == 100+10*x+y);
            BOOST_CHECK(im1.view().at(x,y) == 100+10*x+y);
            BOOST_CHECK(im2.at(x,y) == 100+10*x+y);
            BOOST_CHECK(im2_view.at(x,y) == 100+10*x+y);
            BOOST_CHECK(im2_cview.at(x,y) == 100+10*x+y);
            im1.setValue(x,y, 10*x + y);
            im2.setValue(x,y, 10*x + y);
            BOOST_CHECK(im1(x,y) == 10*x+y);
            BOOST_CHECK(im1.view()(x,y) == 10*x+y);
            BOOST_CHECK(im2(x,y) == 10*x+y);
            BOOST_CHECK(im2_view(x,y) == 10*x+y);
            BOOST_CHECK(im2_cview(x,y) == 10*x+y);
        }
    }

    // Check view of given data
    // Note: Our array is on the stack, so we don't have any ownership to pass around.
    //       Hence, use a default shared_ptr constructor.
    galsim::ImageView<T> im3_view(ref_array, boost::shared_ptr<T>(), ncol, bounds, 1.);
    galsim::ConstImageView<T> im3_cview(ref_array, boost::shared_ptr<T>(), ncol, bounds, 1.);
    for (int y=1; y<=nrow; ++y) {
        for (int x=1; x<=ncol; ++x) {
            BOOST_CHECK(im3_view(x,y) == 10*x+y);
            BOOST_CHECK(im3_cview(x,y) == 10*x+y);
        }
    }

    // Check shift ops
    int dx = 31;
    int dy = 16;
    im1.shift(dx,dy);
    im2_view.setOrigin( 1+dx , 1+dy );
    im3_cview.setCenter( (ncol+1)/2+dx , (nrow+1)/2+dy );
    galsim::Bounds<int> shifted_bounds(1+dx, ncol+dx, 1+dy, nrow+dy);

    BOOST_CHECK(im1.getBounds() == shifted_bounds);
    BOOST_CHECK(im2_view.getBounds() == shifted_bounds);
    BOOST_CHECK(im3_cview.getBounds() == shifted_bounds);
    // Others shouldn't have changed.
    BOOST_CHECK(im2.getBounds() == bounds);
    BOOST_CHECK(im2_cview.getBounds() == bounds);
    BOOST_CHECK(im3_view.getBounds() == bounds);
    for (int y=1; y<=nrow; ++y) {
        for (int x=1; x<=ncol; ++x) {
            BOOST_CHECK(im1(x+dx,y+dy) == 10*x+y);
            BOOST_CHECK(im2(x,y) == 10*x+y);
            BOOST_CHECK(im2_view(x+dx,y+dy) == 10*x+y);
            BOOST_CHECK(im2_cview(x,y) == 10*x+y);
            BOOST_CHECK(im3_view(x,y) == 10*x+y);
            BOOST_CHECK(im3_cview(x+dx,y+dy) == 10*x+y);
        }
    }
}

BOOST_AUTO_TEST_CASE_TEMPLATE( TestImageArith , T , test_types )
{
    const int ncol=7;
    const int nrow=5;
    T ref_array[nrow*ncol] = {
        11, 21, 31, 41, 51, 61, 71,
        12, 22, 32, 42, 52, 62, 72,
        13, 23, 33, 43, 53, 63, 73,
        14, 24, 34, 44, 54, 64, 74,
        15, 25, 35, 45, 55, 65, 75 };
    galsim::Bounds<int> bounds(1,ncol,1,nrow);

    galsim::ConstImageView<T> ref_im(ref_array, boost::shared_ptr<T>(), ncol, bounds, 1.);

    galsim::Image<T> im1 = ref_im;
    galsim::Image<T> im2 = T(2) * ref_im;
    for (int y=1; y<=nrow; ++y) {
        for (int x=1; x<=ncol; ++x) {
            BOOST_CHECK(im2(x,y) == 2 * ref_im(x,y));
        }
    }

    // Test image addition
    { 
        galsim::Image<T> im3 = im1 + im2;
        BOOST_CHECK(im3.getBounds() == bounds);
        for (int y=1; y<=nrow; ++y) {
            for (int x=1; x<=ncol; ++x) {
                BOOST_CHECK(im3(x,y) == 3 * ref_im(x,y));
            }
        }
        im3.fill(0);
        im3.view() = im1 + im2;
        for (int y=1; y<=nrow; ++y) {
            for (int x=1; x<=ncol; ++x) {
                BOOST_CHECK(im3(x,y) == 3 * ref_im(x,y));
            }
        }
        im3 += im2;
        for (int y=1; y<=nrow; ++y) {
            for (int x=1; x<=ncol; ++x) {
                BOOST_CHECK(im3(x,y) == 5 * ref_im(x,y));
            }
        }
        im3.view() += im2;
        for (int y=1; y<=nrow; ++y) {
            for (int x=1; x<=ncol; ++x) {
                BOOST_CHECK(im3(x,y) == 7 * ref_im(x,y));
            }
        }
    }

    // Test image subtraction
    { 
        galsim::Image<T> im3 = im1 - im2;
        BOOST_CHECK(im3.getBounds() == bounds);
        for (int y=1; y<=nrow; ++y) {
            for (int x=1; x<=ncol; ++x) {
                BOOST_CHECK(im3(x,y) == -ref_im(x,y));
            }
        }
        im3.fill(0);
        im3.view() = im1 - im2;
        for (int y=1; y<=nrow; ++y) {
            for (int x=1; x<=ncol; ++x) {
                BOOST_CHECK(im3(x,y) == -ref_im(x,y));
            }
        }
        im3 -= im2;
        for (int y=1; y<=nrow; ++y) {
            for (int x=1; x<=ncol; ++x) {
                BOOST_CHECK(im3(x,y) == -3 * ref_im(x,y));
            }
        }
        im3.view() -= im2;
        for (int y=1; y<=nrow; ++y) {
            for (int x=1; x<=ncol; ++x) {
                BOOST_CHECK(im3(x,y) == -5 * ref_im(x,y));
            }
        }
    }

    // Test binary multiplication
    { 
        galsim::Image<T> im3 = im1 * im2;
        BOOST_CHECK(im3.getBounds() == bounds);
        for (int y=1; y<=nrow; ++y) {
            for (int x=1; x<=ncol; ++x) {
                BOOST_CHECK(im3(x,y) == 2 * ref_im(x,y) * ref_im(x,y));
            }
        }
        im3.fill(0);
        im3.view() = im1 * im2;
        for (int y=1; y<=nrow; ++y) {
            for (int x=1; x<=ncol; ++x) {
                BOOST_CHECK(im3(x,y) == 2 * ref_im(x,y) * ref_im(x,y));
                im3(x,y) /= ref_im(x,y);
            }
        }
        im3 *= im2;
        for (int y=1; y<=nrow; ++y) {
            for (int x=1; x<=ncol; ++x) {
                BOOST_CHECK(im3(x,y) == 4 * ref_im(x,y) * ref_im(x,y));
                im3(x,y) /= 2 * ref_im(x,y);
            }
        }
        im3.view() *= im2;
        for (int y=1; y<=nrow; ++y) {
            for (int x=1; x<=ncol; ++x) {
                // Note: 8 * ref_im(x,y) * ref_im(x,y) exceeds the maximum value for short
                // but 4 * ref_im(x,y) * ref_im(x,y) is ok for ref_im(7,5) = 75
                BOOST_CHECK(im3(x,y) == 4 * ref_im(x,y) * ref_im(x,y));
            }
        }
    }

    // Test binary division
    { 
        for (int y=1; y<=nrow; ++y) {
            for (int x=1; x<=ncol; ++x) {
                im1(x,y) = 4 * ref_im(x,y) * ref_im(x,y);
            }
        }
        galsim::Image<T> im3 = im1 / im2;
        BOOST_CHECK(im3.getBounds() == bounds);
        for (int y=1; y<=nrow; ++y) {
            for (int x=1; x<=ncol; ++x) {
                BOOST_CHECK(im3(x,y) == 2 * ref_im(x,y));
            }
        }
        im3.fill(0);
        im3.view() = im1 / im2;
        for (int y=1; y<=nrow; ++y) {
            for (int x=1; x<=ncol; ++x) {
                BOOST_CHECK(im3(x,y) == 2 * ref_im(x,y));
                im3(x,y) *= ref_im(x,y);
            }
        }
        im3 /= im2;
        for (int y=1; y<=nrow; ++y) {
            for (int x=1; x<=ncol; ++x) {
                BOOST_CHECK(im3(x,y) == ref_im(x,y));
                im3(x,y) *= 4 * ref_im(x,y);
            }
        }
        im3.view() /= im2;
        for (int y=1; y<=nrow; ++y) {
            for (int x=1; x<=ncol; ++x) {
                BOOST_CHECK(im3(x,y) == 2 * ref_im(x,y));
            }
        }
        im1 = ref_im;
    }

    // Test image scalar addition
    { 
        galsim::Image<T> im3 = im1 + T(3);
        BOOST_CHECK(im3.getBounds() == bounds);
        for (int y=1; y<=nrow; ++y) {
            for (int x=1; x<=ncol; ++x) {
                BOOST_CHECK(im3(x,y) == ref_im(x,y) + 3);
            }
        }
        im3.fill(0);
        im3.view() = im1 + T(3);
        for (int y=1; y<=nrow; ++y) {
            for (int x=1; x<=ncol; ++x) {
                BOOST_CHECK(im3(x,y) == ref_im(x,y) + 3);
            }
        }
        im3 += T(3);
        for (int y=1; y<=nrow; ++y) {
            for (int x=1; x<=ncol; ++x) {
                BOOST_CHECK(im3(x,y) == ref_im(x,y) + 6);
            }
        }
        im3.view() += T(3);
        for (int y=1; y<=nrow; ++y) {
            for (int x=1; x<=ncol; ++x) {
                BOOST_CHECK(im3(x,y) == ref_im(x,y) + 9);
            }
        }
    }

    // Test image subtraction
    { 
        galsim::Image<T> im3 = im1 - T(3);
        BOOST_CHECK(im3.getBounds() == bounds);
        for (int y=1; y<=nrow; ++y) {
            for (int x=1; x<=ncol; ++x) {
                BOOST_CHECK(im3(x,y) == ref_im(x,y) - 3);
            }
        }
        im3.fill(0);
        im3.view() = im1 - T(3);
        for (int y=1; y<=nrow; ++y) {
            for (int x=1; x<=ncol; ++x) {
                BOOST_CHECK(im3(x,y) == ref_im(x,y) - 3);
            }
        }
        im3 -= T(3);
        for (int y=1; y<=nrow; ++y) {
            for (int x=1; x<=ncol; ++x) {
                BOOST_CHECK(im3(x,y) == ref_im(x,y) - 6);
            }
        }
        im3.view() -= T(3);
        for (int y=1; y<=nrow; ++y) {
            for (int x=1; x<=ncol; ++x) {
                BOOST_CHECK(im3(x,y) == ref_im(x,y) - 9);
            }
        }
    }

    // Test binary multiplication
    { 
        galsim::Image<T> im3 = im1 * T(3);
        BOOST_CHECK(im3.getBounds() == bounds);
        for (int y=1; y<=nrow; ++y) {
            for (int x=1; x<=ncol; ++x) {
                BOOST_CHECK(im3(x,y) == ref_im(x,y) * 3);
            }
        }
        im3.fill(0);
        im3.view() = im1 * T(3);
        for (int y=1; y<=nrow; ++y) {
            for (int x=1; x<=ncol; ++x) {
                BOOST_CHECK(im3(x,y) == ref_im(x,y) * 3);
            }
        }
        im3 *= T(3);
        for (int y=1; y<=nrow; ++y) {
            for (int x=1; x<=ncol; ++x) {
                BOOST_CHECK(im3(x,y) == ref_im(x,y) * 9);
            }
        }
        im3.view() *= T(3);
        for (int y=1; y<=nrow; ++y) {
            for (int x=1; x<=ncol; ++x) {
                BOOST_CHECK(im3(x,y) == ref_im(x,y) * 27);
            }
        }
    }

    // Test binary division
    { 
        for (int y=1; y<=nrow; ++y) {
            for (int x=1; x<=ncol; ++x) {
                im1(x,y) = ref_im(x,y) * 27;
            }
        }
        galsim::Image<T> im3 = im1 / T(3);
        BOOST_CHECK(im3.getBounds() == bounds);
        for (int y=1; y<=nrow; ++y) {
            for (int x=1; x<=ncol; ++x) {
                BOOST_CHECK(std::fabs(im3(x,y) - ref_im(x,y) * 9) < 0.0001);
            }
        }
        im3.fill(0);
        im3.view() = im1 / T(3);
        for (int y=1; y<=nrow; ++y) {
            for (int x=1; x<=ncol; ++x) {
                BOOST_CHECK(std::fabs(im3(x,y) - ref_im(x,y) * 9) < 0.0001);
            }
        }
        im3 /= T(3);
        for (int y=1; y<=nrow; ++y) {
            for (int x=1; x<=ncol; ++x) {
                BOOST_CHECK(std::fabs(im3(x,y) - ref_im(x,y) * 3) < 0.0001);
            }
        }
        im3.view() /= T(3);
        for (int y=1; y<=nrow; ++y) {
            for (int x=1; x<=ncol; ++x) {
                BOOST_CHECK(std::fabs(im3(x,y) - ref_im(x,y)) < 0.0001);
            }
        }
        im1 = ref_im;
    }
}


BOOST_AUTO_TEST_SUITE_END();
