/* -*- c++ -*-
 * Copyright (c) 2012-2021 by the GalSim developers team on GitHub
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

#include <complex>
#include "galsim/Image.h"

namespace galsim {

    // These ops are not defined by the standard library, but might be required below.
    inline std::complex<double> operator*(double x, const std::complex<float>& y)
    { return std::complex<double>(x*y.real(), x*y.imag()); }
    inline std::complex<double> operator/(double x, const std::complex<float>& y)
    { return x * (1.F / y); }
    inline std::complex<float>& operator+=(std::complex<float>& x, const std::complex<double>& y)
    { return x += std::complex<float>(y); }

    //
    // Templates for stepping through image pixels
    // Not all of these are used for the below arithmetic, but we keep them all
    // here anyway.
    //

    /**
     *  @brief Call a unary function on each pixel value
     */
    template <typename T, typename Op>
    void for_each_pixel_ref(const BaseImage<T>& image, Op& f)
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
    }

    template <typename T, typename Op>
    void for_each_pixel(const BaseImage<T>& image, Op f)
    { for_each_pixel_ref(image, f); }

    /**
     *  @brief Call a function of (value, i, j) on each pixel value
     */
    template <typename T, typename Op>
    void for_each_pixel_ij_ref(const BaseImage<T>& image, Op& f)
    {
        const T* ptr = image.getData();
        if (ptr) {
            const int skip = image.getNSkip();
            const int step = image.getStep();
            const int xmin = image.getXMin();
            const int xmax = image.getXMax();
            const int ymin = image.getYMin();
            const int ymax = image.getYMax();
            if (step == 1) {
                for (int j=ymin; j<=ymax; j++, ptr+=skip)
                    for (int i=xmin; i<=xmax; i++) f(*ptr++,i,j);
            } else {
                for (int j=ymin; j<=ymax; j++, ptr+=skip)
                    for (int i=xmin; i<=xmax; i++, ptr+=step) f(*ptr,i,j);
            }
        }
    }

    template <typename T, typename Op>
    void for_each_pixel_ij(const BaseImage<T>& image, Op f)
    { for_each_pixel_ij_ref(image, f); }

    /**
     *  @brief Replace image with a function of its pixel values.
     */
    template <typename T, typename Op>
    void transform_pixel_ref(ImageView<T> image, Op& f)
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
    }

    template <typename T, typename Op>
    void transform_pixel(ImageView<T> image, Op f)
    { transform_pixel_ref(image, f); }

    /**
     *  @brief Assign function of 2 images to 1st
     */
    template <typename T1, typename T2, typename Op>
    void transform_pixel_ref(ImageView<T1> image1, const BaseImage<T2>& image2, Op& f)
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
                    for (int i=0; i<ncol; i++, ++ptr1, ++ptr2) *ptr1 = f(*ptr1,T1(*ptr2));
            } else {
                for (int j=0; j<nrow; j++, ptr1+=skip1, ptr2+=skip2)
                    for (int i=0; i<ncol; i++, ptr1+=step1, ptr2+=step2) *ptr1 = f(*ptr1,T1(*ptr2));
            }
        }
    }

    template <typename T1, typename T2, typename Op>
    void transform_pixel(ImageView<T1> image1, const BaseImage<T2>& image2, Op f)
    { transform_pixel_ref(image1, image2, f); }

    // Some functionals that are useful for operating on images:
    template <typename T>
    class ConstReturn
    {
    public:
        ConstReturn(const T v): val(v) {}
        inline T operator()(const T ) const { return val; }
    private:
        T val;
    };

    template <typename T>
    class ReturnInverse
    {
    public:
        inline T operator()(const T val) const { return val==T(0) ? T(0.) : T(1./val); }
    };

    template <typename T, typename T2>
    class AddConstant
    {
        const T2 _x;
    public:
        AddConstant(const T2 x) : _x(x) {}
        inline T operator()(const T val) const { return T(_x + val); }
    };

    template <typename T, typename T2>
    class MultiplyConstant
    {
        const T2 _x;
    public:
        MultiplyConstant(const T2 x) : _x(x) {}
        inline T operator()(const T val) const { return T(_x * val); }
    };

    template <typename T, typename T2, bool is_int>
    class DivideConstant // is_int=False
    {
        const T2 _invx;
    public:
        DivideConstant(const T2 x) : _invx(T2(1)/x) {}
        inline T operator()(const T val) const { return T(val * _invx); }
    };

    template <typename T, typename T2>
    class DivideConstant<T,T2,true>
    {
        const T2 _x;
    public:
        DivideConstant(const T2 x) : _x(x) {}
        inline T operator()(const T val) const { return T(val / _x); }
    };

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
    struct ResultType<uint32_t,double> { typedef double type; };
    template <>
    struct ResultType<uint16_t,double> { typedef double type; };

    template <>
    struct ResultType<int32_t,float> { typedef float type; };
    template <>
    struct ResultType<int16_t,float> { typedef float type; };
    template <>
    struct ResultType<uint32_t,float> { typedef float type; };
    template <>
    struct ResultType<uint16_t,float> { typedef float type; };

    template <>
    struct ResultType<int16_t,int32_t> { typedef int32_t type; };
    template <>
    struct ResultType<uint16_t,uint32_t> { typedef uint32_t type; };

    // For convenience below...
#define CT std::complex<T>

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
        void assignTo(ImageView<result_type> rhs) const { rhs = _im; rhs += _x; }
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
    // Update: we do allow arithmetic between complex images and their corresponding real scalar.
    template <typename T>
    inline SumIX<T,T> operator+(const BaseImage<T>& im, T x)
    { return SumIX<T,T>(im,x); }

    template <typename T>
    inline SumIX<T,T> operator+(T x, const BaseImage<T>& im)
    { return SumIX<T,T>(im,x); }

    template <typename T>
    inline ImageView<T> operator+=(ImageView<T> im, T x)
    { transform_pixel(im, AddConstant<T,T>(x)); return im; }

    template <typename T>
    inline ImageAlloc<T>& operator+=(ImageAlloc<T>& im, const T& x)
    { im.view() += x; return im; }

    template <typename T>
    inline SumIX<CT,T> operator+(const BaseImage<CT>& im, T x)
    { return SumIX<CT,T>(im,x); }

    template <typename T>
    inline SumIX<CT,T> operator+(T x, const BaseImage<CT>& im)
    { return SumIX<CT,T>(im,x); }

    template <typename T>
    inline ImageView<CT> operator+=(ImageView<CT> im, T x)
    { transform_pixel(im, AddConstant<CT,T>(x)); return im; }

    template <typename T>
    inline ImageAlloc<CT>& operator+=(ImageAlloc<CT>& im, const T& x)
    { im.view() += x; return im; }


    //
    // Image - Scalar
    //

    template <typename T>
    inline SumIX<T,T> operator-(const BaseImage<T>& im, T x)
    { return SumIX<T,T>(im,-x); }

    template <typename T>
    inline ImageView<T> operator-=(ImageView<T> im, const T& x)
    { im += T(-x); return im; }

    template <typename T>
    inline ImageAlloc<T>& operator-=(ImageAlloc<T>& im, const T& x)
    { im.view() -= x; return im; }

    template <typename T>
    inline SumIX<CT,T> operator-(const BaseImage<CT>& im, T x)
    { return SumIX<CT,T>(im,-x); }

    template <typename T>
    inline ImageView<CT> operator-=(ImageView<CT> im, T x)
    { im += T(-x); return im; }

    template <typename T>
    inline ImageAlloc<CT>& operator-=(ImageAlloc<CT>& im, const T& x)
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
        void assignTo(ImageView<result_type> rhs) const { rhs = _im; rhs *= _x; }
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
    inline ImageView<T> operator*=(ImageView<T> im, const T& x)
    { transform_pixel(im, MultiplyConstant<T,T>(x)); return im; }

    template <typename T>
    inline ImageAlloc<T>& operator*=(ImageAlloc<T>& im, const T& x)
    { im.view() *= x; return im; }

    template <typename T>
    inline ProdIX<CT,T> operator*(const BaseImage<CT>& im, T x)
    { return ProdIX<CT,T>(im,x); }

    template <typename T>
    inline ProdIX<CT,T> operator*(T x, const BaseImage<CT>& im)
    { return ProdIX<CT,T>(im,x); }

    template <typename T>
    inline ImageView<CT> operator*=(ImageView<CT> im, T x)
    { transform_pixel(im, MultiplyConstant<CT,T>(x)); return im; }

    template <typename T>
    inline ImageAlloc<CT>& operator*=(ImageAlloc<CT>& im, const T& x)
    { im.view() *= x; return im; }

    // Specialize variants that can be sped up using SSE
    PUBLIC_API ImageView<float> operator*=(ImageView<float> im, float x);
    PUBLIC_API ImageView<std::complex<float> > operator*=(
        ImageView<std::complex<float> > im, float x);
    PUBLIC_API ImageView<std::complex<float> > operator*=(
        ImageView<std::complex<float> > im, std::complex<float> x);
    PUBLIC_API ImageView<double> operator*=(ImageView<double> im, double x);
    PUBLIC_API ImageView<std::complex<double> > operator*=(
        ImageView<std::complex<double> > im, double x);
    PUBLIC_API ImageView<std::complex<double> > operator*=(
        ImageView<std::complex<double> > im, std::complex<double> x);


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
        void assignTo(ImageView<result_type> rhs) const { rhs = _im; rhs /= _x; }
    private:
        const BaseImage<T1>& _im;
        const T2 _x;
    };

    template <typename T>
    inline QuotIX<T,T> operator/(const BaseImage<T>& im, T x)
    { return QuotIX<T,T>(im,x); }

#define INT(T) std::numeric_limits<T>::is_integer
    template <typename T>
    inline ImageView<T> operator/=(ImageView<T> im, T x)
    { transform_pixel(im, DivideConstant<T,T,INT(T)>(x)); return im; }

    template <typename T>
    inline ImageAlloc<T>& operator/=(ImageAlloc<T>& im, const T& x)
    { im.view() /= x; return im; }

    template <typename T>
    inline QuotIX<CT,T> operator/(const BaseImage<CT>& im, T x)
    { return QuotIX<CT,T>(im,x); }

    template <typename T>
    inline ImageView<CT> operator/=(ImageView<CT> im, T x)
    { transform_pixel(im, DivideConstant<CT,T,INT(T)>(x)); return im; }

    template <typename T>
    inline ImageAlloc<CT>& operator/=(ImageAlloc<CT>& im, const T& x)
    { im.view() /= x; return im; }

#undef CT

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
        void assignTo(ImageView<result_type> rhs) const { rhs = _im1; rhs += _im2; }
    private:
        const BaseImage<T1>& _im1;
        const BaseImage<T2>& _im2;
    };

    template <typename T1, typename T2>
    inline SumII<T1,T2> operator+(const BaseImage<T1>& im1, const BaseImage<T2>& im2)
    { return SumII<T1,T2>(im1,im2); }

    template <typename T1, typename T2>
    inline ImageView<T1> operator+=(ImageView<T1> im1, const BaseImage<T2>& im2)
    {
        if (!im1.getBounds().isSameShapeAs(im2.getBounds()))
            throw ImageError("Attempt im1 += im2, but bounds not the same shape");
        transform_pixel(im1, im2, std::plus<T1>());
        return im1;
    }

    template <typename T1, typename T2>
    inline ImageAlloc<T1>& operator+=(ImageAlloc<T1>& im, const BaseImage<T2>& im2)
    { im.view() += im2; return im; }


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
        void assignTo(ImageView<result_type> rhs) const { rhs = _im1; rhs -= _im2; }
    private:
        const BaseImage<T1>& _im1;
        const BaseImage<T2>& _im2;
    };

    template <typename T1, typename T2>
    inline DiffII<T1,T2> operator-(const BaseImage<T1>& im1, const BaseImage<T2>& im2)
    { return DiffII<T1,T2>(im1,im2); }

    template <typename T1, typename T2>
    inline ImageView<T1> operator-=(ImageView<T1> im1, const BaseImage<T2>& im2)
    {
        if (!im1.getBounds().isSameShapeAs(im2.getBounds()))
            throw ImageError("Attempt im1 -= im2, but bounds not the same shape");
        transform_pixel(im1, im2, std::minus<T1>());
        return im1;
    }

    template <typename T1, typename T2>
    inline ImageAlloc<T1>& operator-=(ImageAlloc<T1>& im, const BaseImage<T2>& im2)
    { im.view() -= im2; return im; }


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
        void assignTo(ImageView<result_type> rhs) const { rhs = _im1; rhs *= _im2; }
    private:
        const BaseImage<T1>& _im1;
        const BaseImage<T2>& _im2;
    };

    template <typename T1, typename T2>
    inline ProdII<T1,T2> operator*(const BaseImage<T1>& im1, const BaseImage<T2>& im2)
    { return ProdII<T1,T2>(im1,im2); }

    template <typename T1, typename T2>
    inline ImageView<T1> operator*=(ImageView<T1> im1, const BaseImage<T2>& im2)
    {
        if (!im1.getBounds().isSameShapeAs(im2.getBounds()))
            throw ImageError("Attempt im1 *= im2, but bounds not the same shape");
        transform_pixel(im1, im2, std::multiplies<T1>());
        return im1;
    }

    template <typename T1, typename T2>
    inline ImageAlloc<T1>& operator*=(ImageAlloc<T1>& im, const BaseImage<T2>& im2)
    { im.view() *= im2; return im; }

    // Specialize variants that can be sped up using SSE
    PUBLIC_API ImageView<float> operator*=(ImageView<float> im1, const BaseImage<float>& im2);
    PUBLIC_API ImageView<std::complex<float> > operator*=(
        ImageView<std::complex<float> > im1, const BaseImage<float>& im2);
    PUBLIC_API ImageView<std::complex<float> > operator*=(
        ImageView<std::complex<float> > im1, const BaseImage<std::complex<float> >& im2);

    PUBLIC_API ImageView<double> operator*=(ImageView<double> im1, const BaseImage<double>& im2);
    PUBLIC_API ImageView<std::complex<double> > operator*=(
        ImageView<std::complex<double> > im1, const BaseImage<double>& im2);
    PUBLIC_API ImageView<std::complex<double> > operator*=(
        ImageView<std::complex<double> > im1, const BaseImage<std::complex<double> >& im2);


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
        void assignTo(ImageView<result_type> rhs) const { rhs = _im1; rhs /= _im2; }
    private:
        const BaseImage<T1>& _im1;
        const BaseImage<T2>& _im2;
    };

    template <typename T1, typename T2>
    inline QuotII<T1,T2> operator/(const BaseImage<T1>& im1, const BaseImage<T2>& im2)
    { return QuotII<T1,T2>(im1,im2); }

    template <typename T1, typename T2>
    inline ImageView<T1> operator/=(ImageView<T1> im1, const BaseImage<T2>& im2)
    {
        if (!im1.getBounds().isSameShapeAs(im2.getBounds()))
            throw ImageError("Attempt im1 /= im2, but bounds not the same shape");
        transform_pixel(im1, im2, std::divides<T1>());
        return im1;
    }

    template <typename T1, typename T2>
    inline ImageAlloc<T1>& operator/=(ImageAlloc<T1>& im, const BaseImage<T2>& im2)
    { im.view() /= im2; return im; }

    //! @endcond

} // namespace galsim

#endif
