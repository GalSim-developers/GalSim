/* -*- c++ -*-
 * Copyright (c) 2012-2016 by the GalSim developers team on GitHub
 * https://github.com/GalSim-developers
 *
 * This file is part of GalSim: The modular galaxy image simulation toolkit.
 * https://github.com/GalSim-developers/GalSim
 *
 * GalSim is free software: redistribution and use in source and binary forms,
 * with or without modification, are permitted provided that the following
 * conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions, and the disclaimer given in the accompanying LICENSE
 *    file.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions, and the disclaimer given in the documentation
 *    and/or other materials provided with the distribution.
 */

#ifndef GalSim_ImageArith_H
#define GalSim_ImageArith_H

namespace galsim {

    // All code between the @cond and @endcond is excluded from Doxygen documentation
    //! @cond

    /**
     *  @brief Exception class usually thrown by images.
     */
    class ImageError : public std::runtime_error {
    public:
        ImageError(const std::string& m) : std::runtime_error("Image Error: " + m) {}

    };

    /**
     *  @brief Exception class thrown when out-of-bounds pixels are accessed on an image.
     */
    class ImageBoundsError : public ImageError {
    public:
        ImageBoundsError(const std::string& m) :
            ImageError("Access to out-of-bounds pixel " + m) {}

        ImageBoundsError(const std::string& m, int min, int max, int tried);

        ImageBoundsError(int x, int y, const Bounds<int> b);
    };

    //! @endcond


    template <typename T> class AssignableToImage;
    template <typename T> class BaseImage;
    template <typename T> class ImageAlloc;
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
    Op for_each_pixel(const BaseImage<T>& image, Op f)
    {
        const T* ptr = image.getData();
        if (ptr) {
            const int skip = image.getNSkip();
            const int step = image.getStep();
            const int nrow = image.getNRow();
            const int ncol = image.getNCol();
            if (step == 1) {
                for (int j=0; j<nrow; j++, ptr+=skip)
                    for (int i=0; i<ncol; i++) f(*ptr++);
            } else {
                for (int j=0; j<nrow; j++, ptr+=skip)
                    for (int i=0; i<ncol; i++, ptr+=step) f(*ptr);
            }
        }
        return f;
    }

    /**
     *  @brief Replace image with a function of its pixel values.
     */
    template <typename T, typename Op>
    Op transform_pixel(const ImageView<T>& image, Op f)
    {
        T* ptr = image.getData();
        if (ptr) {
            const int skip = image.getNSkip();
            const int step = image.getStep();
            const int nrow = image.getNRow();
            const int ncol = image.getNCol();
            if (step == 1) {
                for (int j=0; j<nrow; j++, ptr+=skip)
                    for (int i=0; i<ncol; i++, ++ptr) *ptr = f(*ptr);
            } else {
                for (int j=0; j<nrow; j++, ptr+=skip)
                    for (int i=0; i<ncol; i++, ptr+=step) *ptr = f(*ptr);
            }
        }
        return f;
    }

    /**
     *  @brief Assign function of 2 images to 1st
     */
    template <typename T1, typename T2, typename Op>
    Op transform_pixel(const ImageView<T1>& image1, const BaseImage<T2>& image2, Op f)
    {
        T1* ptr1 = image1.getData();
        if (ptr1) {

            if (!image1.getBounds().isSameShapeAs(image2.getBounds()))
                throw ImageError("transform_pixel image bounds are not same shape");

            const int skip1 = image1.getNSkip();
            const int step1 = image1.getStep();
            const int nrow = image1.getNRow();
            const int ncol = image1.getNCol();
            const T2* ptr2 = image2.getData();
            const int skip2 = image2.getNSkip();
            const int step2 = image2.getStep();
            if (step1 == 1 && step2 == 1) {
                for (int j=0; j<nrow; j++, ptr1+=skip1, ptr2+=skip2)
                    for (int i=0; i<ncol; i++, ++ptr1, ++ptr2) *ptr1 = f(*ptr1,*ptr2);
            } else {
                for (int j=0; j<nrow; j++, ptr1+=skip1, ptr2+=skip2)
                    for (int i=0; i<ncol; i++, ptr1+=step1, ptr2+=step2) *ptr1 = f(*ptr1,*ptr2);
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
    inline ImageAlloc<T>& operator+=(ImageAlloc<T>& im, const T& x)
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
    inline ImageAlloc<T>& operator-=(ImageAlloc<T>& im, const T& x)
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
    inline ImageAlloc<T>& operator*=(ImageAlloc<T>& im, const T& x)
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
    inline ImageAlloc<T>& operator/=(ImageAlloc<T>& im, const T& x)
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
    inline ImageAlloc<T1>& operator+=(ImageAlloc<T1>& im, const BaseImage<T2>& x)
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
    inline ImageAlloc<T1>& operator-=(ImageAlloc<T1>& im, const BaseImage<T2>& x)
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
    inline ImageAlloc<T1>& operator*=(ImageAlloc<T1>& im, const BaseImage<T2>& x)
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
    inline ImageAlloc<T1>& operator/=(ImageAlloc<T1>& im, const BaseImage<T2>& x)
    { im.view() /= x; return im; }

    //! @endcond

} // namespace galsim

#endif
