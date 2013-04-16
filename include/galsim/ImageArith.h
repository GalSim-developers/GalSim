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

#ifndef ImageArith_H
#define ImageArith_H

namespace galsim {

    // All code between the @cond and @endcond is excluded from Doxygen documentation
    //! @cond

    /**
     *  @brief Exception class usually thrown by images.
     */
    class ImageError : public std::runtime_error {
    public: 
        ImageError(const std::string& m="") : 
            std::runtime_error("Image Error: " + m) {}

    };

    /**
     *  @brief Exception class thrown when out-of-bounds pixels are accessed on an image.
     */
    class ImageBoundsError : public ImageError {
    public: 
        ImageBoundsError(const std::string& m="") : 
            ImageError("Access to out-of-bounds pixel " + m) {}

        ImageBoundsError(const std::string& m, int min, int max, int tried);

        ImageBoundsError(int x, int y, const Bounds<int> b);
    };

    //! @endcond


    template <typename T> class AssignableToImage;
    template <typename T> class BaseImage;
    template <typename T> class Image;
    template <typename T> class ImageView;

    //
    // Templates for stepping through image pixels
    // Not all of these are used for the below arithmetic, but we keep them all
    // here anyway.
    //

    /**
     *  @brief Call a unary function on each pixel value
     */
    template <typename T, typename Op>
    Op for_each_pixel(const ImageView<T>& image, Op f) 
    {
        // Note: all of these functions have this guard to make sure we don't
        // try to access the memory if the image is in an undefined state.
        if (image.getData()) {
            if (image.isContiguous()) {
                f = std::for_each(image.rowBegin(image.getYMin()), image.rowEnd(image.getYMax()), f);
            } else {
                for (int i = image.getYMin(); i <= image.getYMax(); i++)
                    f = std::for_each(image.rowBegin(i), image.rowEnd(i), f);
            }
        }
        return f;
    }

    /**
     *  @brief Call a unary function on each pixel in a subset of the image.
     */
    template <typename T, typename Op>
    Op for_each_pixel(const ImageView<T>& image, const Bounds<int>& bounds, Op f) 
    {
        if (image.getData()) {
            if (!image.getBounds().includes(bounds))
                throw ImageError("for_each_pixel range exceeds image range");

            if (image.getBounds() == bounds) return for_each_pixel(image,f);

            for (int i = bounds.getYMin(); i <= bounds.getYMax(); i++)
                f = std::for_each(image.getIter(bounds.getXMin(),i),
                                  image.getIter(bounds.getXMax()+1,i), f);
        }
        return f;
    }

    /**
     *  @brief Replace image with a function of its pixel values.
     */
    template <typename T, typename Op>
    Op transform_pixel(const ImageView<T>& image, Op f) 
    {
        if (image.getData()) {
            typedef typename ImageView<T>::iterator Iter;
            if (image.isContiguous()) {
                const Iter ee = image.rowEnd(image.getYMax());
                for (Iter it = image.rowBegin(image.getYMin()); it != ee; ++it) 
                    *it = f(*it);
            } else {
                for (int y = image.getYMin(); y <= image.getYMax(); ++y) {
                    const Iter ee = image.rowEnd(y);
                    for (Iter it = image.rowBegin(y); it != ee; ++it) 
                        *it = f(*it);
                }
            }
        }
        return f;
    }

    /** 
     *  @brief Replace a subset of the image with a function of its pixel values.
     */
    template <typename T, typename Op>
    Op transform_pixel(const ImageView<T>& image, const Bounds<int>& bounds, Op f) 
    {
        if (image.getData()) {
            typedef typename ImageView<T>::iterator Iter;
            if (!image.getBounds().includes(bounds))
                throw ImageError("transform_pixel range exceeds image range");

            if (image.getBounds() == bounds) return transform_pixel(image,f); 

            for (int y = bounds.getYMin(); y <= bounds.getYMax(); ++y) {
                const Iter ee = image.getIter(bounds.getXMax()+1,y);      
                for (Iter it = image.getIter(bounds.getXMin(),y); it != ee; ++it) 
                    *it = f(*it);
            }
        }
        return f;
    }

    /**
     *  @brief Add a function of pixel coords to an image.
     */
    template <typename T, typename Op>
    Op add_function_pixel(const ImageView<T>& image, Op f) 
    {
        if (image.getData()) {
            typedef typename ImageView<T>::iterator Iter;
            for (int y = image.getYMin(); y <= image.getYMax(); ++y) {
                int x = image.getXMin();
                const Iter ee = image.rowEnd(y);
                for (Iter it = image.rowBegin(y); it != ee; ++it, ++x) 
                    *it += f(x,y);
            }
        }
        return f;
    }

    /**
     *  @brief Add a function of pixel coords to a subset of an image.
     */
    template <typename T, typename Op>
    Op add_function_pixel(const ImageView<T>& image, const Bounds<int>& bounds, Op f) 
    {
        if (image.getData()) {
            typedef typename ImageView<T>::iterator Iter;
            if (!bounds.isDefined()) return f;
            if (!image.getBounds().includes(bounds))
                throw ImageError("add_function_pixel range exceeds image range");

            if (image.getBounds() == bounds) return add_function_pixel(image,f);

            for (int y = bounds.getYMin(); y <= bounds.getYMax(); ++y) {
                int x = bounds.getXMin();
                const Iter ee = image.getIter(bounds.getXMax()+1,y);      
                for (Iter it = image.getIter(bounds.getXMin(),y); it != ee; ++it, ++x) 
                    *it += f(x,y);
            }
        }
        return f;
    }

    /**
     *  @brief Replace image with a function of pixel coords.
     */
    template <typename T, typename Op>
    Op fill_pixel(const ImageView<T>& image, Op f) 
    {
        if (image.getData()) {
            typedef typename ImageView<T>::iterator Iter;
            for (int y = image.getYMin(); y <= image.getYMax(); ++y) {
                int x = image.getXMin();
                const Iter ee = image.rowEnd(y);      
                for (Iter it = image.rowBegin(y); it != ee; ++it, ++x) 
                    *it = f(x,y);
            }
        }
        return f;
    }

    /**
     *  @brief Replace subset of an image with a function of pixel coords.
     */
    template <typename T, typename Op>
    Op fill_pixel(const ImageView<T>& image, const Bounds<int>& bounds, Op f) 
    {
        if (image.getData()) {
            typedef typename ImageView<T>::iterator Iter;
            if (!image.getBounds().includes(bounds))
                throw ImageError("add_function_pixel range exceeds image range");

            if (image.getBounds() == bounds) return fill_pixel(image,f);

            for (int y = bounds.getYMin(); y <= bounds.getYMax(); ++y) {
                int x = bounds.getXMin();
                const Iter ee = image.getIter(bounds.getXMax()+1,y);      
                for (Iter it = image.getIter(bounds.getXMin(),y); it != ee; ++it, ++x) 
                    *it = f(x,y);
            }
        }
        return f;
    }

    // Assign function of 2 images to 1st
    template <typename T1, typename T2, typename Op>
    Op transform_pixel(const ImageView<T1>& image1, const BaseImage<T2>& image2, Op f) 
    {
        if (image1.getData()) {
            typedef typename ImageView<T1>::iterator Iter1;
            typedef typename BaseImage<T2>::const_iterator Iter2;

            if (!image1.getBounds().isSameShapeAs(image2.getBounds()))
                throw ImageError("transform_pixel image bounds are not same shape");

            int y2 = image2.getYMin();
            for (int y = image1.getYMin(); y <= image1.getYMax(); ++y, ++y2) {
                Iter2 it2 = image2.rowBegin(y2);
                const Iter1 ee = image1.rowEnd(y);      
                for (Iter1 it1 = image1.rowBegin(y); it1 != ee; ++it1, ++it2)
                    *it1 = f(*it1,*it2);
            }
        }
        return f;
    }

    // Assign function of Img2 & Img3 to Img1
    template <typename T1, typename T2, typename T3, typename Op>
    Op transform_pixel(
        const ImageView<T1>& image1,
        const BaseImage<T2>& image2, 
        const BaseImage<T3>& image3,
        Op f) 
    {
        if (image1.getData()) {
            typedef typename ImageView<T1>::iterator Iter1;
            typedef typename BaseImage<T2>::const_iterator Iter2;
            typedef typename BaseImage<T3>::const_iterator Iter3;

            if (!image1.getBounds().isSameShapeAs(image2.getBounds()))
                throw ImageError("transform_pixel image1, image2 bounds are not same shape");
            if (!image1.getBounds().isSameShapeAs(image3.getBounds()))
                throw ImageError("transform_pixel image1, image3 bounds are not same shape");

            int y2 = image2.getYMin();
            int y3 = image3.getYMin();
            for (int y = image1.getYMin(); y <= image1.getYMax(); ++y, ++y2, ++y3) {
                Iter2 it2 = image2.rowBegin(y2);
                Iter3 it3 = image3.rowBegin(y3);
                const Iter1 ee = image1.rowEnd(y);      
                for (Iter1 it1 = image1.rowBegin(y); it1 != ee; ++it1, ++it2, ++it3) 
                    *it1 = f(*it2,*it3);
            }
        }
        return f;
    }

    // Assign function of 2 images to 1st over bounds
    // Note that for this one, the two images do not have to be the same shape.
    // They just both have to include the given bounds.
    // Only that portion of each image is used for the calculation.
    template <typename T1, typename T2, typename Op>
    Op transform_pixel(
        const ImageView<T1>& image1,
        const BaseImage<T2>& image2,
        const Bounds<int>& bounds,
        Op f) 
    {
        if (image1.getData()) {
            typedef typename ImageView<T1>::iterator Iter1;
            typedef typename Image<T2>::iterator Iter2;
            if (!image1.getBounds().includes(bounds) || !image2.getBounds().includes(bounds))
                throw ImageError("transform_pixel range exceeds image range");

            if (image1.getBounds() == bounds && image2.getBounds() == bounds) 
                return transform_pixel(image1,image2,f);

            for (int y = bounds.getYMin(); y <= bounds.getYMax(); ++y) {
                const Iter1 ee = image1.getIter(bounds.getXMax()+1,y);      
                Iter2 it2 = image2.getIter(bounds.getXMin(),y);
                for (Iter1 it1 = image1.getIter(bounds.getXMin(),y); it1 != ee; ++it1, ++it2) 
                    *it1 = f(*it1,*it2);
            }
        }
        return f;
    }

    // All code between the @cond and @endcond is excluded from Doxygen documentation
    //! @cond

    // Default uses T1 as the result type
    template <typename T1, typename T2>
    struct ResultType { typedef T1 type; };

    // Specialize relevant cases where T2 should be the result type
    template <>
    struct ResultType<float,double> { typedef double type; };
    template <>
    struct ResultType<int32_t,double> { typedef double type; };
    template <>
    struct ResultType<int16_t,double> { typedef double type; };

    template <>
    struct ResultType<int32_t,float> { typedef float type; };
    template <>
    struct ResultType<int16_t,float> { typedef float type; };

    template <>
    struct ResultType<int16_t,int32_t> { typedef int32_t type; };

    //
    // Image + Scalar
    //

    template <typename T1, typename T2> 
    class SumIX : public AssignableToImage<typename ResultType<T1,T2>::type>
    {
    public:
        typedef typename ResultType<T1,T2>::type result_type;
        SumIX(const BaseImage<T1>& im, const T2 x) :
            AssignableToImage<result_type>(im.getBounds()), _im(im), _x(x) {}
        void assignTo(const ImageView<result_type>& rhs) const { rhs = _im; rhs += _x; }
    private:
        const BaseImage<T1>& _im;
        const T2 _x;
    };

    // Currently, the only valid type for x is the value type of im.
    // The reason is that making the type of x a template opens the door
    // to any class, not just POD.  I don't know of an easy way to make 
    // this only valid for T2 = POD types like int16_t, int32_t, float, double.
    // So if we do want to allow mixed types for these, we'd probably have to 
    // specifically overload each one by hand.
    template <typename T>
    inline SumIX<T,T> operator+(const BaseImage<T>& im, T x) 
    { return SumIX<T,T>(im,x); }

    template <typename T>
    inline SumIX<T,T> operator+(T x, const BaseImage<T>& im) 
    { return SumIX<T,T>(im,x); }

    template <typename T> 
    inline const ImageView<T>& operator+=(const ImageView<T>& im, T x) 
    { transform_pixel(im, bind2nd(std::plus<T>(),x)); return im; }

    template <typename T>
    inline Image<T>& operator+=(Image<T>& im, const T& x) 
    { im.view() += x; return im; }


    //
    // Image - Scalar
    //

    template <typename T>
    inline SumIX<T,T> operator-(const BaseImage<T>& im, T x) 
    { return SumIX<T,T>(im,-x); }

    template <typename T> 
    inline const ImageView<T>& operator-=(const ImageView<T>& im, T x) 
    { im += T(-x); return im; }

    template <typename T>
    inline Image<T>& operator-=(Image<T>& im, const T& x) 
    { im.view() -= x; return im; }


    //
    // Image * Scalar
    //

    template <typename T1, typename T2> 
    class ProdIX : public AssignableToImage<typename ResultType<T1,T2>::type>
    {
    public:
        typedef typename ResultType<T1,T2>::type result_type;
        ProdIX(const BaseImage<T1>& im, const T2 x) :
            AssignableToImage<result_type>(im.getBounds()), _im(im), _x(x) {}
        void assignTo(const ImageView<result_type>& rhs) const { rhs = _im; rhs *= _x; }
    private:
        const BaseImage<T1>& _im;
        const T2 _x;
    };

    template <typename T>
    inline ProdIX<T,T> operator*(const BaseImage<T>& im, T x) 
    { return ProdIX<T,T>(im,x); }

    template <typename T>
    inline ProdIX<T,T> operator*(T x, const BaseImage<T>& im) 
    { return ProdIX<T,T>(im,x); }

    template <typename T> 
    inline const ImageView<T>& operator*=(const ImageView<T>& im, T x) 
    { transform_pixel(im, bind2nd(std::multiplies<T>(),x)); return im; }

    template <typename T>
    inline Image<T>& operator*=(Image<T>& im, const T& x) 
    { im.view() *= x; return im; }

    //
    // Image / Scalar
    //

    template <typename T1, typename T2> 
    class QuotIX : public AssignableToImage<typename ResultType<T1,T2>::type>
    {
    public:
        typedef typename ResultType<T1,T2>::type result_type;
        QuotIX(const BaseImage<T1>& im, const T2 x) :
            AssignableToImage<result_type>(im.getBounds()), _im(im), _x(x) {}
        void assignTo(const ImageView<result_type>& rhs) const { rhs = _im; rhs /= _x; }
    private:
        const BaseImage<T1>& _im;
        const T2 _x;
    };

    template <typename T>
    inline QuotIX<T,T> operator/(const BaseImage<T>& im, T x) 
    { return QuotIX<T,T>(im,x); }

    template <typename T>
    inline QuotIX<T,T> operator/(T x, const BaseImage<T>& im) 
    { return QuotIX<T,T>(im,x); }

    template <typename T> 
    inline const ImageView<T>& operator/=(const ImageView<T>& im, T x) 
    { transform_pixel(im, bind2nd(std::divides<T>(),x)); return im; }

    template <typename T>
    inline Image<T>& operator/=(Image<T>& im, const T& x) 
    { im.view() /= x; return im; }

    //
    // Image + Image
    //

    template <typename T1, typename T2> 
    class SumII : public AssignableToImage<typename ResultType<T1,T2>::type>
    {
    public:
        typedef typename ResultType<T1,T2>::type result_type;
        SumII(const BaseImage<T1>& im1, const BaseImage<T2>& im2) :
            AssignableToImage<result_type>(im1.getBounds()), _im1(im1), _im2(im2) 
        {
            if (!im1.getBounds().isSameShapeAs(im2.getBounds()))
                throw ImageError("Attempt im1 + im2, but bounds not the same shape");
        }
        void assignTo(const ImageView<result_type>& rhs) const { rhs = _im1; rhs += _im2; }
    private:
        const BaseImage<T1>& _im1;
        const BaseImage<T2>& _im2;
    };

    template <typename T1, typename T2>
    inline SumII<T1,T2> operator+(const BaseImage<T1>& im1, const BaseImage<T2>& im2) 
    { return SumII<T1,T2>(im1,im2); }

    template <typename T1, typename T2> 
    inline const ImageView<T1>& operator+=(const ImageView<T1>& im1, const BaseImage<T2>& im2) 
    {
        if (!im1.getBounds().isSameShapeAs(im2.getBounds()))
            throw ImageError("Attempt im1 += im2, but bounds not the same shape");
        transform_pixel(im1, im2, std::plus<T1>());
        return im1; 
    }

    template <typename T1, typename T2> 
    inline Image<T1>& operator+=(Image<T1>& im, const BaseImage<T2>& x) 
    { im.view() += x; return im; }


    //
    // Image - Image
    //

    template <typename T1, typename T2> 
    class DiffII : public AssignableToImage<typename ResultType<T1,T2>::type>
    {
    public:
        typedef typename ResultType<T1,T2>::type result_type;
        DiffII(const BaseImage<T1>& im1, const BaseImage<T2>& im2) :
            AssignableToImage<result_type>(im1.getBounds()), _im1(im1), _im2(im2) 
        {
            if (!im1.getBounds().isSameShapeAs(im2.getBounds()))
                throw ImageError("Attempt im1 - im2, but bounds not the same shape");
        }
        void assignTo(const ImageView<result_type>& rhs) const { rhs = _im1; rhs -= _im2; }
    private:
        const BaseImage<T1>& _im1;
        const BaseImage<T2>& _im2;
    };

    template <typename T1, typename T2>
    inline DiffII<T1,T2> operator-(const BaseImage<T1>& im1, const BaseImage<T2>& im2) 
    { return DiffII<T1,T2>(im1,im2); }

    template <typename T1, typename T2> 
    inline const ImageView<T1>& operator-=(const ImageView<T1>& im1, const BaseImage<T2>& im2) 
    {
        if (!im1.getBounds().isSameShapeAs(im2.getBounds()))
            throw ImageError("Attempt im1 -= im2, but bounds not the same shape");
        transform_pixel(im1, im2, std::minus<T1>());
        return im1; 
    }

    template <typename T1, typename T2> 
    inline Image<T1>& operator-=(Image<T1>& im, const BaseImage<T2>& x) 
    { im.view() -= x; return im; }


    //
    // Image * Image
    //

    template <typename T1, typename T2> 
    class ProdII : public AssignableToImage<typename ResultType<T1,T2>::type>
    {
    public:
        typedef typename ResultType<T1,T2>::type result_type;
        ProdII(const BaseImage<T1>& im1, const BaseImage<T2>& im2) :
            AssignableToImage<result_type>(im1.getBounds()), _im1(im1), _im2(im2) 
        {
            if (!im1.getBounds().isSameShapeAs(im2.getBounds()))
                throw ImageError("Attempt im1 * im2, but bounds not the same shape");
        }
        void assignTo(const ImageView<result_type>& rhs) const { rhs = _im1; rhs *= _im2; }
    private:
        const BaseImage<T1>& _im1;
        const BaseImage<T2>& _im2;
    };

    template <typename T1, typename T2>
    inline ProdII<T1,T2> operator*(const BaseImage<T1>& im1, const BaseImage<T2>& im2) 
    { return ProdII<T1,T2>(im1,im2); }

    template <typename T1, typename T2>
    inline const ImageView<T1>& operator*=(const ImageView<T1>& im1, const BaseImage<T2>& im2) 
    {
        if (!im1.getBounds().isSameShapeAs(im2.getBounds()))
            throw ImageError("Attempt im1 *= im2, but bounds not the same shape");
        transform_pixel(im1, im2, std::multiplies<T1>());
        return im1; 
    }

    template <typename T1, typename T2>
    inline Image<T1>& operator*=(Image<T1>& im, const BaseImage<T2>& x) 
    { im.view() *= x; return im; }


    //
    // Image / Image
    //

    template <typename T1, typename T2> 
    class QuotII : public AssignableToImage<typename ResultType<T1,T2>::type>
    {
    public:
        typedef typename ResultType<T1,T2>::type result_type;
        QuotII(const BaseImage<T1>& im1, const BaseImage<T2>& im2) :
            AssignableToImage<result_type>(im1.getBounds()), _im1(im1), _im2(im2) 
        {
            if (!im1.getBounds().isSameShapeAs(im2.getBounds()))
                throw ImageError("Attempt im1 / im2, but bounds not the same shape");
        }
        void assignTo(const ImageView<result_type>& rhs) const { rhs = _im1; rhs /= _im2; }
    private:
        const BaseImage<T1>& _im1;
        const BaseImage<T2>& _im2;
    };

    template <typename T1, typename T2>
    inline QuotII<T1,T2> operator/(const BaseImage<T1>& im1, const BaseImage<T2>& im2) 
    { return QuotII<T1,T2>(im1,im2); }

    template <typename T1, typename T2>
    inline const ImageView<T1>& operator/=(const ImageView<T1>& im1, const BaseImage<T2>& im2) 
    {
        if (!im1.getBounds().isSameShapeAs(im2.getBounds()))
            throw ImageError("Attempt im1 /= im2, but bounds not the same shape");
        transform_pixel(im1, im2, std::divides<T1>());
        return im1; 
    }

    template <typename T1, typename T2>
    inline Image<T1>& operator/=(Image<T1>& im, const BaseImage<T2>& x) 
    { im.view() /= x; return im; }

    //! @endcond

} // namespace galsim

#endif
