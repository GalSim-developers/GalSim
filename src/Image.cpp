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

//#define DEBUGLOGGING

#include <sstream>
#include <numeric>
#include <cstring>

#include "fftw3.h"
#include "fmath/fmath.hpp"  // Use their compiler checks for the right SSE include.

#include "Image.h"
#include "ImageArith.h"

namespace galsim {


// A helper class to let us write things like REAL(x) or CONJ(x) for real or complex.
template <typename T>
struct ComplexHelper
{
    typedef T real_type;
    typedef std::complex<T> complex_type;
    static inline T conj(const T& x) { return x; }
    static inline T real(const T& x) { return x; }
};
template <typename T>
struct ComplexHelper<std::complex<T> >
{
    typedef T real_type;
    typedef std::complex<T> complex_type;
    static inline T real(const std::complex<T>& x) { return std::real(x); }
    static inline std::complex<T> conj(const std::complex<T>& x) { return std::conj(x); }
};
template <typename T>
inline typename ComplexHelper<T>::real_type REAL(const T& x) { return ComplexHelper<T>::real(x); }
template <typename T>
inline T CONJ(const T& x) { return ComplexHelper<T>::conj(x); }


/////////////////////////////////////////////////////////////////////
//// Constructor for out-of-bounds that has coordinate info
///////////////////////////////////////////////////////////////////////


std::string MakeErrorMessage(
    const std::string& m, const int min, const int max, const int tried)
{
    // See discussion in Std.h about this initial value.
    std::ostringstream oss(" ");
    oss << "Attempt to access "<<m<<" number "<<tried
        << ", range is "<<min<<" to "<<max;
    return oss.str();
}
ImageBoundsError::ImageBoundsError(
    const std::string& m, const int min, const int max, const int tried) :
    ImageError(MakeErrorMessage(m,min,max,tried))
{}

std::string MakeErrorMessage(const int x, const int y, const Bounds<int> b)
{
    std::ostringstream oss(" ");
    bool found=false;
    if (x<b.getXMin() || x>b.getXMax()) {
        oss << "Attempt to access column number "<<x
            << ", range is "<<b.getXMin()<<" to "<<b.getXMax();
        found = true;
    }
    if (y<b.getYMin() || y>b.getYMax()) {
        if (found) oss << " and ";
        oss << "Attempt to access row number "<<y
            << ", range is "<<b.getYMin()<<" to "<<b.getYMax();
        found = true;
    }
    if (!found) return "Cannot find bounds violation ???";
    else return oss.str();
}
ImageBoundsError::ImageBoundsError(const int x, const int y, const Bounds<int> b) :
    ImageError(MakeErrorMessage(x,y,b))
{}

/////////////////////////////////////////////////////////////////////
//// Constructor (and related helpers) for the various Image classes
///////////////////////////////////////////////////////////////////////

template <typename T>
BaseImage<T>::BaseImage(const Bounds<int>& b) :
    AssignableToImage<T>(b), _owner(), _data(0), _nElements(0), _step(0), _stride(0),
    _ncol(0), _nrow(0)
{
    if (this->_bounds.isDefined()) allocateMem();
    // Else _data is left as 0, step,stride = 0.
}

// A custom deleter that finds the original address of the memory allocation directly
// before the stored pointer and frees that using delete []
template <typename T>
struct AlignedDeleter {
    void operator()(T* p) const { delete [] ((char**)p)[-1]; }
};

template <typename T>
std::shared_ptr<T> allocateAlignedMemory(int n)
{
    // This bit is based on the answers here:
    // http://stackoverflow.com/questions/227897/how-to-allocate-aligned-memory-only-using-the-standard-library/227900
    // The point of this is to get the _data pointer aligned to a 16 byte (128 bit) boundary.
    // Arrays that are so aligned can use SSE operations and so can be much faster than
    // non-aligned memroy.  FFTW in particular is faster if it gets aligned data.
    char* mem = new char[n * sizeof(T) + sizeof(char*) + 15];
    T* data = reinterpret_cast<T*>( (uintptr_t)(mem + sizeof(char*) + 15) & ~(size_t) 0x0F );
    ((char**)data)[-1] = mem;
    std::shared_ptr<T> owner(data, AlignedDeleter<T>());
    return owner;
}

template <typename T>
void BaseImage<T>::allocateMem()
{
    // Note: this version always does the memory (re-)allocation.
    // So the various functions that call this should do their (different) checks
    // for whether this is necessary.
    _step = 1;
    _stride = _ncol = this->_bounds.getXMax() - this->_bounds.getXMin() + 1;
    _nrow = this->_bounds.getYMax() - this->_bounds.getYMin() + 1;

    _nElements = _stride * (this->_bounds.getYMax() - this->_bounds.getYMin() + 1);
    if (_stride <= 0 || _nElements <= 0) {
        FormatAndThrow<ImageError>() <<
            "Attempt to create an Image with defined but invalid Bounds ("<<this->_bounds<<")";
    }

    _owner = allocateAlignedMemory<T>(_nElements);
    _data = _owner.get();
}

template <typename T>
struct Sum
{
    Sum(): sum(0) {}
    void operator()(T x) { sum += double(x); }
    double sum;
};

template <typename T>
struct Sum<std::complex<T> >
{
    Sum(): sum(0) {}
    void operator()(std::complex<T> x) { sum += std::complex<double>(x); }
    std::complex<double> sum;
};

template <typename T>
T BaseImage<T>::sumElements() const
{
    Sum<T> sum;
    for_each_pixel_ref(*this, sum);
    return T(sum.sum);
}

// Avoid warning about abs being a no op for unsigned types
template <typename T>
inline typename Traits<T>::real_type ABS(T x)
{ return std::abs(x); }
template <> inline uint32_t ABS(uint32_t x) { return x; }
template <> inline uint16_t ABS(uint16_t x) { return x; }

template <typename T>
struct MaxAbs
{
    MaxAbs(): max(0) {}
    void operator()(T x) { T absx = ABS(x); if (absx > max) max = absx; }
    T max;
};

template <typename T>
struct MaxAbs<std::complex<T> >
{
    typedef typename Traits<T>::real_type RT;
    MaxAbs(): max(0) {}
    void operator()(std::complex<T> x) { RT absx = ABS(x); if (absx > max) max = absx; }
    RT max;
};

template <typename T>
typename Traits<T>::real_type BaseImage<T>::maxAbsElement() const
{
    MaxAbs<T> max;
    for_each_pixel_ref(*this, max);
    return (max.max);
}

template <typename T>
struct NonZeroBounds
{
    NonZeroBounds(): bounds() {}
    void operator()(T x, int i, int j) { if (x != T(0)) bounds += Position<int>(i,j); }
    Bounds<int> bounds;
};

template <typename T>
Bounds<int> BaseImage<T>::nonZeroBounds() const
{
    NonZeroBounds<T> nz;
    for_each_pixel_ij_ref(*this, nz);
    return nz.bounds;
}

void CheckSize(int ncol, int nrow)
{
    if (ncol <= 0 || nrow <= 0) {
        std::ostringstream oss(" ");
        if (ncol <= 0) {
            if (nrow <= 0) {
                oss << "Attempt to create an Image with non-positive ncol ("<<
                    ncol<<") and nrow ("<<nrow<<")";
            } else {
                oss << "Attempt to create an Image with non-positive ncol ("<<
                    ncol<<")";
            }
        } else {
            oss << "Attempt to create an Image with non-positive nrow ("<<
                nrow<<")";
        }
        throw ImageError(oss.str());
    }
}

template <typename T>
ImageAlloc<T>::ImageAlloc(int ncol, int nrow) :
    BaseImage<T>(Bounds<int>(1,ncol,1,nrow))
{
    CheckSize(ncol, nrow);
#ifdef DEBUGLOGGING
    fill(T(99999));
#endif
}

template <typename T>
ImageAlloc<T>::ImageAlloc(int ncol, int nrow, T init_value) :
    BaseImage<T>(Bounds<int>(1,ncol,1,nrow))
{
    CheckSize(ncol, nrow);
    fill(init_value);
}

template <typename T>
ImageAlloc<T>::ImageAlloc(const Bounds<int>& bounds) :
    BaseImage<T>(bounds)
{
#ifdef DEBUGLOGGING
    fill(T(99999));
#endif
}

template <typename T>
ImageAlloc<T>::ImageAlloc(const Bounds<int>& bounds, const T init_value) :
    BaseImage<T>(bounds)
{
    fill(init_value);
}

template <typename T>
void ImageAlloc<T>::resize(const Bounds<int>& new_bounds)
{
    if (!new_bounds.isDefined()) {
        // Then this is really a deallocation.  Clear out the existing memory.
        this->_bounds = new_bounds;
        this->_owner.reset();
        this->_data = 0;
        this->_nElements = 0;
        this->_step = 0;
        this->_stride = 0;
        this->_ncol = 0;
        this->_nrow = 0;
    } else if (this->_bounds.isDefined() &&
               new_bounds.area() <= this->_nElements &&
               this->_owner.unique()) {
        // Then safe to keep existing memory allocation.
        // Just redefine the bounds and stride.
        this->_bounds = new_bounds;
        this->_stride = this->_ncol = new_bounds.getXMax() - new_bounds.getXMin() + 1;
        this->_nrow = new_bounds.getYMax() - new_bounds.getYMin() + 1;
    } else {
        // Then we want to do the reallocation.
        this->_bounds = new_bounds;
        this->allocateMem();
    }
}


/////////////////////////////////////////////////////////////////////
//// Access methods
///////////////////////////////////////////////////////////////////////

template <typename T>
const T& BaseImage<T>::at(const int xpos, const int ypos) const
{
    if (!_data) throw ImageError("Attempt to access values of an undefined image");
    if (!this->_bounds.includes(xpos, ypos)) throw ImageBoundsError(xpos, ypos, this->_bounds);
    return _data[addressPixel(xpos, ypos)];
}

template <typename T>
T& ImageView<T>::at(const int xpos, const int ypos)
{
    if (!this->_data) throw ImageError("Attempt to access values of an undefined image");
    if (!this->_bounds.includes(xpos, ypos)) throw ImageBoundsError(xpos, ypos, this->_bounds);
    return this->_data[this->addressPixel(xpos, ypos)];
}

template <typename T>
T& ImageAlloc<T>::at(const int xpos, const int ypos)
{
    if (!this->_data) throw ImageError("Attempt to access values of an undefined image");
    if (!this->_bounds.includes(xpos, ypos)) throw ImageBoundsError(xpos, ypos, this->_bounds);
    return this->_data[this->addressPixel(xpos, ypos)];
}

template <typename T>
const T& ImageAlloc<T>::at(const int xpos, const int ypos) const
{
    if (!this->_data) throw ImageError("Attempt to access values of an undefined image");
    if (!this->_bounds.includes(xpos, ypos)) throw ImageBoundsError(xpos, ypos, this->_bounds);
    return this->_data[this->addressPixel(xpos, ypos)];
}

template <typename T>
ConstImageView<T> BaseImage<T>::subImage(const Bounds<int>& bounds) const
{
    if (!_data) throw ImageError("Attempt to make subImage of an undefined image");
    if (!this->_bounds.includes(bounds)) {
        FormatAndThrow<ImageError>() <<
            "Subimage bounds (" << bounds << ") are outside original image bounds (" <<
            this->_bounds << ")";
    }
    T* newdata = _data
        + (bounds.getYMin() - this->_bounds.getYMin()) * _stride
        + (bounds.getXMin() - this->_bounds.getXMin()) * _step;
    return ConstImageView<T>(newdata,_owner,_step,_stride,bounds);
}

template <typename T>
ImageView<T> ImageView<T>::subImage(const Bounds<int>& bounds)
{
    if (!this->_data) throw ImageError("Attempt to make subImage of an undefined image");
    if (!this->_bounds.includes(bounds)) {
        FormatAndThrow<ImageError>() <<
            "Subimage bounds (" << bounds << ") are outside original image bounds (" <<
            this->_bounds << ")";
    }
    T* newdata = this->_data
        + (bounds.getYMin() - this->_bounds.getYMin()) * this->_stride
        + (bounds.getXMin() - this->_bounds.getXMin()) * this->_step;
    return ImageView<T>(newdata,this->_owner,this->_step,this->_stride,bounds);
}


// Some helper functions to make the logic in the below wrap function easier to follow

// Add row j to row jj
// ptr and ptrwrap should be at the start of the respective rows.
// At the end of this function, each will be one past the end of the row.
template <typename T>
void wrap_row(T*& ptr, T*& ptrwrap, int m, int step)
{
    // Add contents of row j to row jj
    if (step == 1)
        for(; m; --m) *ptrwrap++ += *ptr++;
    else
        for(; m; --m,ptr+=step,ptrwrap+=step) *ptrwrap += *ptr;
}

template <typename T>
void wrap_cols(T*& ptr, int m, int mwrap, int i1, int i2, int step)
{
    int ii = i2 - (i2 % mwrap);
    if (ii == i2) ii = i1;
    T* ptrwrap = ptr + ii*step;
    // First do i in [0,i1).
    for(int i=0; i<i1;) {
        xdbg<<"Start loop at i = "<<i<<std::endl;
        // How many do we do before looping back
        int k = i2-ii;
        xdbg<<"k = "<<k<<std::endl;
        if (step == 1)
            for (; k; --k, ++i) *ptrwrap++ += *ptr++;
        else
            for (; k; --k, ++i, ptr+=step, ptrwrap+=step) *ptrwrap += *ptr;
        ii = i1;
        ptrwrap -= mwrap*step;
    }
    // Skip ahead to do i in [i2,m)
    assert(ii == i1);
    assert(ptr == ptrwrap);
    ptr += mwrap * step;
    for(int i=i2; i<m;) {
        xdbg<<"Start loop at i = "<<i<<std::endl;
        // How many do we do before looping back or ending.
        int k = std::min(m-i, mwrap);
        xdbg<<"k = "<<k<<std::endl;
        if (step == 1)
            for (; k; --k, ++i) *ptrwrap++ += *ptr++;
        else
            for (; k; --k, ++i, ptr+=step, ptrwrap+=step) *ptrwrap += *ptr;
        ptrwrap -= mwrap*step;
    }
}

// Add conjugate of row j to row jj
// ptrwrap should be at the end of the conjugate row, not the beginning.
// At the end of this function, ptr will be one past the end of the row, and ptrskip will be
// one before the beginning.
template <typename T>
void wrap_row_conj(T*& ptr, T*& ptrwrap, int m, int step)
{
    if (step == 1)
        for(; m; --m) *ptrwrap-- += CONJ(*ptr++);
    else
        for(; m; --m,ptr+=step,ptrwrap-=step) *ptrwrap += CONJ(*ptr);
}

// If j == jj, this needs to be slightly different.
template <typename T>
void wrap_row_selfconj(T*& ptr, T*& ptrwrap, int m, int step)
{
    if (step == 1)
        for(int i=(m+1)/2; i; --i,++ptr,--ptrwrap) {
            *ptrwrap += CONJ(*ptr);
            *ptr = CONJ(*ptrwrap);
        }
    else
        for(int i=(m+1)/2; i; --i,ptr+=step,ptrwrap-=step) {
            *ptrwrap += CONJ(*ptr);
            *ptr = CONJ(*ptrwrap);
        }
    ptr += (m-(m+1)/2) * step;
    ptrwrap -= (m-(m+1)/2) * step;
}

// Wrap two half-rows where one has the conjugate information for the other.
// ptr1 and ptr2 should start at the the pointer for i=mwrap within the two rows.
// At the end of this function, they will each be one past the end of the rows.
template <typename T>
void wrap_hermx_cols_pair(T*& ptr1, T*& ptr2, int m, int mwrap, int step)
{
    // We start the wrapping with col N/2 (aka i2-1), which needs to wrap its conjugate
    // (-N/2) onto itself.
    // Then as i increases, we decrease ii and continue wrapping conjugates.
    // When we get to something that wraps onto col 0 (the first one will correspond to
    // i=-N, which is the conjugate of what is stored at i=N in the other row), we need to
    // repeat with a regular non-conjugate wrapping for the positive col (e.g. i=N,j=j itself)
    // which also wraps onto col 0.
    // Then we run ii back up wrapping normally, until we get to N/2 again (aka i2-1).
    // The negative col will wrap normally onto -N/2, which means we need to also do a
    // conjugate wrapping onto N/2.

    xdbg<<"Start hermx_cols_pair\n";
    T* ptr1wrap = ptr1;
    T* ptr2wrap = ptr2;
    int i = mwrap-1;
    while (1) {
        xdbg<<"Start loop at i = "<<i<<std::endl;
        // Do the first column with a temporary to avoid overwriting.
        T temp = *ptr1;
        *ptr1wrap += CONJ(*ptr2);
        *ptr2wrap += CONJ(temp);
        ptr1 += step;
        ptr2 += step;
        ptr1wrap -= step;
        ptr2wrap -= step;
        ++i;
        // Progress as normal (starting at i=mwrap for the first loop).
        int k = std::min(m-i, mwrap-2);
        xdbg<<"k = "<<k<<std::endl;
        if (step == 1)
            for (; k; --k, ++i) {
                *ptr1wrap-- += CONJ(*ptr2++);
                *ptr2wrap-- += CONJ(*ptr1++);
            }
        else
            for (; k; --k, ++i, ptr1+=step, ptr2+=step, ptr1wrap-=step, ptr2wrap-=step) {
                *ptr1wrap += CONJ(*ptr2);
                *ptr2wrap += CONJ(*ptr1);
            }
        xdbg<<"i = "<<i<<std::endl;
        if (i == m) break;
        // On the last one, don't increment ptrs, since we need to repeat with the non-conj add.
        *ptr1wrap += CONJ(*ptr2);
        *ptr2wrap += CONJ(*ptr1);
        k = std::min(m-i, mwrap-1);
        xdbg<<"k = "<<k<<std::endl;
        if (step == 1)
            for (; k; --k, ++i) {
                *ptr1wrap++ += *ptr1++;
                *ptr2wrap++ += *ptr2++;
            }
        else
            for (; k; --k, ++i, ptr1+=step, ptr2+=step, ptr1wrap+=step, ptr2wrap+=step) {
                *ptr1wrap += *ptr1;
                *ptr2wrap += *ptr2;
            }
        xdbg<<"i = "<<i<<std::endl;
        if (i == m) break;
        *ptr1wrap += *ptr1;
        *ptr2wrap += *ptr2;
    }
}

// Wrap a single half-row that is its own conjugate (i.e. j==0)
template <typename T>
void wrap_hermx_cols(T*& ptr, int m, int mwrap, int step)
{
    xdbg<<"Start hermx_cols\n";
    T* ptrwrap = ptr;
    int i = mwrap-1;
    while (1) {
        xdbg<<"Start loop at i = "<<i<<std::endl;
        int k = std::min(m-i, mwrap-1);
        xdbg<<"k = "<<k<<std::endl;
        if (step == 1)
            for (; k; --k, ++i) *ptrwrap-- += CONJ(*ptr++);
        else
            for (; k; --k, ++i, ptr+=step, ptrwrap-=step) *ptrwrap += CONJ(*ptr);
        xdbg<<"i = "<<i<<std::endl;
        if (i == m) break;
        *ptrwrap += CONJ(*ptr);
        k = std::min(m-i, mwrap-1);
        xdbg<<"k = "<<k<<std::endl;
        if (step == 1)
            for (; k; --k, ++i) *ptrwrap++ += *ptr++;
        else
            for (; k; --k, ++i, ptr+=step, ptrwrap+=step) *ptrwrap += *ptr;
        xdbg<<"i = "<<i<<std::endl;
        if (i == m) break;
        *ptrwrap += *ptr;
    }
}

template <typename T>
void wrapImage(ImageView<T> im, const Bounds<int>& b, bool hermx, bool hermy)
{
    dbg<<"Start ImageView::wrap: b = "<<b<<std::endl;
    dbg<<"self bounds = "<<im.getBounds()<<std::endl;
    //set_verbose(2);

    const int i1 = b.getXMin()-im.getBounds().getXMin();
    const int i2 = b.getXMax()-im.getBounds().getXMin()+1;  // +1 for "1 past the end"
    const int j1 = b.getYMin()-im.getBounds().getYMin();
    const int j2 = b.getYMax()-im.getBounds().getYMin()+1;
    xdbg<<"i1,i2,j1,j2 = "<<i1<<','<<i2<<','<<j1<<','<<j2<<std::endl;
    const int mwrap = i2-i1;
    const int nwrap = j2-j1;
    const int skip = im.getNSkip();
    const int step = im.getStep();
    const int stride = im.getStride();
    const int m = im.getNCol();
    const int n = im.getNRow();
    T* ptr = im.getData();

    if (hermx) {
        // In the hermitian x case, we need to wrap the columns first, otherwise the bookkeeping
        // becomes difficult.
        //
        // Each row has a corresponding row that stores the conjugate information for the
        // negative x values that are not stored.  We do these pairs of rows together.
        //
        // The exception is row 0 (which here is j==(n-1)/2), which is its own conjugate, so
        // it works slightly differently.
        assert(i1 == 0);

        int mid = (n-1)/2;  // The value of j that corresponds to the j==0 in the normal notation.

        T* ptr1 = im.getData() + (i2-1)*step;
        T* ptr2 = im.getData() + (n-1)*stride + (i2-1)*step;

        // These skips will take us from the end of one row to the i2-1 element in the next row.
        int skip1 = skip + (i2-1)*step;
        int skip2 = skip1 - 2*stride; // This is negative.  We add this value to ptr2.

        for (int j=0; j<mid; ++j, ptr1+=skip1, ptr2+=skip2) {
            xdbg<<"Wrap rows "<<j<<","<<n-j-1<<" into columns ["<<i1<<','<<i2<<")\n";
            xdbg<<"ptrs = "<<ptr1-im.getData()<<"  "<<ptr2-im.getData()<<std::endl;
            wrap_hermx_cols_pair(ptr1, ptr2, m, mwrap, step);
        }
        // Finally, the row that is really j=0 (but here is j=(n-1)/2) also needs to be wrapped
        // singly.
        xdbg<<"Wrap row "<<mid<<" into columns ["<<i1<<','<<i2<<")\n";
        xdbg<<"ptrs = "<<ptr1-im.getData()<<"  "<<ptr2-im.getData()<<std::endl;
        wrap_hermx_cols(ptr1, m, mwrap, step);
    }

    // If hermx is false, then we wrap the rows first instead.
    if (hermy) {
        assert(j1 == 0);
        // In this case, the number of rows in the target image corresponds to N/2+1.
        // Rows 0 and N/2 need special handling, since the wrapping is really for the
        // range (-N/2,N/2], even though the negative rows are not stored.
        // We start with row N/2 (aka j2-1), which needs to wrap its conjugate (-N/2) onto itself.
        // Then as j increases, we decrease jj and continue wrapping conjugates.
        // When we get to something that wraps onto row 0 (the first one will correspond to
        // j=-N, which is the conjugate of what is stored at j=N), we need to repeat with
        // a regular non-conjugate wrapping for the positive row (e.g. j=N itself) which also
        // wraps onto row 0.
        // Then we run jj back up wrapping normally, until we get to N/2 again (aka j2-1).
        // The negative row will wrap normally onto -N/2, which means we need to also do a
        // conjugate wrapping onto N/2.

        // Start with j == jj = j2-1.
        int jj = j2-1;
        ptr += jj * stride;
        T* ptrwrap = ptr + (m-1) * step;

        // Do the first row separately, since we need to do it slightly differently, as
        // we are overwriting the input data as we go, so we would double add it if we did
        // it the normal way.
        xdbg<<"Wrap first row "<<jj<<" onto row = "<<jj<<" using conjugation.\n";
        xdbg<<"ptrs = "<<ptr-im.getData()<<"  "<<ptrwrap-im.getData()<<std::endl;
        wrap_row_selfconj(ptr, ptrwrap, m, step);

        ptr += skip;
        ptrwrap -= skip;
        --jj;
        int  j= j2;
        while (1) {
            int k = std::min(n-j,jj);  // How many conjugate rows to do?
            for (; k; --k, ++j, --jj, ptr+=skip, ptrwrap-=skip) {
                xdbg<<"Wrap row "<<j<<" onto row = "<<jj<<" using conjugation.\n";
                xdbg<<"ptrs = "<<ptr-im.getData()<<"  "<<ptrwrap-im.getData()<<std::endl;
                wrap_row_conj(ptr, ptrwrap, m, step);
            }
            assert(j==n || jj == j1);
            if (j == n) break;
            assert(j < n);
            // On the last one, don't increment ptrs, since we need to repeat with the non-conj add.
            wrap_row_conj(ptr, ptrwrap, m, step);
            ptr -= m*step;
            ptrwrap += step;

            k = std::min(n-j,nwrap-1);  // How many non-conjugate rows to do?
            for (; k; --k, ++j, ++jj, ptr+=skip, ptrwrap+=skip) {
                xdbg<<"Wrap row "<<j<<" onto row = "<<jj<<std::endl;
                xdbg<<"ptrs = "<<ptr-im.getData()<<"  "<<ptrwrap-im.getData()<<std::endl;
                wrap_row(ptr, ptrwrap, m, step);
            }
            assert(j==n || jj == j2-1);
            if (j == n) break;
            assert(j < n);
            wrap_row(ptr, ptrwrap, m, step);
            ptr -= m*step;
            ptrwrap -= step;
        }
    } else {
        // The regular case is mostly simpler (no conjugate stuff to worry about).
        // However, we don't have the luxury of knowing that j1==0, so we need to start with
        // the rows j<j1, then skip over [j1,j2) when we get there and continue with j>=j2.

        // Row 0 maps onto j2 - (j2 % nwrap) (although we may need to subtract nwrap).
        int jj = j2 - (j2 % nwrap);
        if (jj == j2) jj = j1;
        T* ptrwrap = ptr + jj * stride;
        for (int j=0; j<n;) {
            // When we get here, we can just skip to j2 and keep going.
            if (j == j1) {
                assert(ptr == ptrwrap);
                j = j2;
                ptr += nwrap * stride;
            }
            int k = std::min(n-j,j2-jj);  // How many to do before looping back.
            for (; k; --k, ++j, ++jj, ptr+=skip, ptrwrap+=skip) {
                xdbg<<"Wrap row "<<j<<" onto row = "<<jj<<std::endl;
                xdbg<<"ptrs = "<<ptr-im.getData()<<"  "<<ptrwrap-im.getData()<<std::endl;
                wrap_row(ptr, ptrwrap, m, step);
            }
            jj = j1;
            ptrwrap -= nwrap * stride;
        }
    }

    // In the normal (not hermx) case, we now wrap rows [j1,j2) into the columns [i1,i2).
    if (!hermx) {
        ptr = im.getData() + j1*stride;
        for (int j=j1; j<j2; ++j, ptr+=skip) {
            xdbg<<"Wrap row "<<j<<" into columns ["<<i1<<','<<i2<<")\n";
            xdbg<<"ptr = "<<ptr-im.getData()<<std::endl;
            wrap_cols(ptr, m, mwrap, i1, i2, step);
        }
    }
}


template <typename T>
void rfft(const BaseImage<T>& in, ImageView<std::complex<double> > out,
          bool shift_in, bool shift_out)
{
    dbg<<"Start rfft\n";
    dbg<<"self bounds = "<<in.getBounds()<<std::endl;

    if (!in.getData() or !in.getBounds().isDefined())
        throw ImageError("Attempting to perform fft on undefined image.");

    const int Nxo2 = in.getBounds().getXMax()+1;
    const int Nyo2 = in.getBounds().getYMax()+1;
    const int Nx = Nxo2 << 1;
    const int Ny = Nyo2 << 1;
    dbg<<"Nx,Ny = "<<Nx<<','<<Ny<<std::endl;

    if (in.getBounds().getYMin() != -Nyo2 || in.getBounds().getXMin() != -Nxo2)
        throw ImageError("fft requires bounds to be (-Nx/2, Nx/2-1, -Ny/2, Ny/2-1)");

    if (out.getBounds().getXMin() != 0 || out.getBounds().getXMax() != Nxo2 ||
        out.getBounds().getYMin() != -Nyo2 || out.getBounds().getYMax() != Nyo2-1)
        throw ImageError("fft requires out.bounds to be (0, Nx/2, -Ny/2, Ny/2-1)");

    if ((uintptr_t) out.getData() % 16 != 0)
        throw ImageError("fft requires out.data to be 16 byte aligned");

    // We will use the same array for input and output.
    // For the input, we just cast the memory to double to use for the input data.
    // However, note that the complex array has two extra elements in the primary direction
    // (x in our case) to allow for the extra column.
    // cf. http://www.fftw.org/doc/Real_002ddata-DFT-Array-Format.html
    double* xptr = reinterpret_cast<double*>(out.getData());
    const T* ptr = in.getData();
    const int skip = in.getNSkip();

    // The FT image that FFTW will return will have FT(0,0) placed at the origin.  We
    // want it placed in the middle instead.  We can make that happen by inverting every other
    // row in the input image.
    if (shift_out) {
        double fac = (shift_in && Nyo2 % 2 == 1) ? -1 : 1.;
        if (in.getStep() == 1) {
            for (int j=Ny; j; --j, ptr+=skip, xptr+=2, fac=-fac)
                for (int i=Nx; i; --i)
                    *xptr++ = fac * REAL(*ptr++);
        } else {
            for (int j=Ny; j; --j, ptr+=skip, xptr+=2, fac=-fac)
                for (int i=Nx; i; --i, ptr+=in.getStep())
                    *xptr++ = fac * REAL(*ptr);
        }
    } else {
        if (in.getStep() == 1) {
            for (int j=Ny; j; --j, ptr+=skip, xptr+=2)
                for (int i=Nx; i; --i)
                    *xptr++ = REAL(*ptr++);
        } else {
            for (int j=Ny; j; --j, ptr+=skip, xptr+=2)
                for (int i=Nx; i; --i, ptr+=in.getStep())
                    *xptr++ = REAL(*ptr);
        }
    }

    fftw_complex* kdata = reinterpret_cast<fftw_complex*>(out.getData());
    double* xdata = reinterpret_cast<double*>(out.getData());

    fftw_plan plan = fftw_plan_dft_r2c_2d(Ny, Nx, xdata, kdata, FFTW_ESTIMATE);
    if (plan==NULL) throw std::runtime_error("fftw_plan cannot be created");
    fftw_execute(plan);
    fftw_destroy_plan(plan);

    // The resulting image will still have a checkerboard pattern of +-1 on it, which
    // we want to remove.
    if (shift_in) {
        std::complex<double>* kptr = out.getData();
        double fac = 1.;
        const bool extra_flip = (Nxo2 % 2 == 1);
        for (int j=Ny; j; --j, fac=(extra_flip?-fac:fac))
            for (int i=Nxo2+1; i; --i, fac=-fac)
                *kptr++ *= fac;
    }
}

template <typename T>
void irfft(const BaseImage<T>& in, ImageView<double> out, bool shift_in, bool shift_out)
{
    dbg<<"Start irfft\n";
    dbg<<"self bounds = "<<in.getBounds()<<std::endl;

    if (!in.getData() or !in.getBounds().isDefined())
        throw ImageError("Attempting to perform inverse fft on undefined image.");

    if (in.getBounds().getXMin() != 0)
        throw ImageError("inverse_fft requires bounds to be (0, Nx/2, -Ny/2, Ny/2-1)");

    const int Nxo2 = in.getBounds().getXMax();
    const int Nyo2 = in.getBounds().getYMax()+1;
    const int Nx = Nxo2 << 1;
    const int Ny = Nyo2 << 1;
    dbg<<"Nx,Ny = "<<Nx<<','<<Ny<<std::endl;

    if (in.getBounds().getYMin() != -Nyo2)
        throw ImageError("inverse_fft requires bounds to be (0, N/2, -N/2, N/2-1)");

    if (out.getBounds().getXMin() != -Nxo2 || out.getBounds().getXMax() != Nxo2+1 ||
        out.getBounds().getYMin() != -Nyo2 || out.getBounds().getYMax() != Nyo2-1)
        throw ImageError("inverse_fft requires out.bounds to be (-Nx/2, Nx/2+1, -Ny/2, Ny/2-1)");

    if ((uintptr_t) out.getData() % 16 != 0)
        throw ImageError("inverse_fft requires out.data to be 16 byte aligned");

    // We will use the same array for input and output.
    // For the input, we just cast the memory to complex<double> to use for the input data.
    // However, note that the real array needs two extra elements in the primary direction
    // (x in our case) to allow for the extra column in the k array.
    // cf. http://www.fftw.org/doc/Real_002ddata-DFT-Array-Format.html
    // The bounds we care about are (-Nxo2, Nxo2-1, -Nyo2, Nyo2-1).

    std::complex<double>* kptr = reinterpret_cast<std::complex<double>*>(out.getData());

    // FFTW wants the locations of the + and - ky values swapped relative to how
    // we store it in an image.
    // Also, to put x=0 in center of array, we need to flop the sign of every other element
    // and need to scale by (1/N)^2.
    double fac = 1./(Nx*Ny);

    const int start_offset = shift_in ? Nyo2 * in.getStride() : 0;
    const int mid_offset = shift_in ? 0 : Nyo2 * in.getStride();

    const int skip = in.getNSkip();
    if (shift_out) {
        const T* ptr = in.getData() + start_offset;
        const bool extra_flip = (Nxo2 % 2 == 1);
        if (in.getStep() == 1) {
            for (int j=Nyo2; j; --j, ptr+=skip, fac=(extra_flip?-fac:fac))
                for (int i=Nxo2+1; i; --i, fac=-fac)
                    *kptr++ = fac * *ptr++;
            ptr = in.getData() + mid_offset;
            for (int j=Nyo2; j; --j, ptr+=skip, fac=(extra_flip?-fac:fac))
                for (int i=Nxo2+1; i; --i, fac=-fac)
                    *kptr++ = fac * *ptr++;
        } else {
            for (int j=Nyo2; j; --j, ptr+=skip, fac=(extra_flip?-fac:fac))
                for (int i=Nxo2+1; i; --i, ptr+=in.getStep(), fac=-fac)
                    *kptr++ = fac * *ptr;
            ptr = in.getData() + mid_offset;
            for (int j=Nyo2; j; --j, ptr+=skip, fac=(extra_flip?-fac:fac))
                for (int i=Nxo2+1; i; --i, ptr+=in.getStep(), fac=-fac)
                    *kptr++ = fac * *ptr;
        }
    } else {
        const T* ptr = in.getData() + start_offset;
        if (in.getStep() == 1) {
            for (int j=Nyo2; j; --j, ptr+=skip)
                for (int i=Nxo2+1; i; --i)
                    *kptr++ = fac * *ptr++;
            ptr = in.getData() + mid_offset;
            for (int j=Nyo2; j; --j, ptr+=skip)
                for (int i=Nxo2+1; i; --i)
                    *kptr++ = fac * *ptr++;
        } else {
            for (int j=Nyo2; j; --j, ptr+=skip)
                for (int i=Nxo2+1; i; --i, ptr+=in.getStep())
                    *kptr++ = fac * *ptr;
            ptr = in.getData() + mid_offset;
            for (int j=Nyo2; j; --j, ptr+=skip)
                for (int i=Nxo2+1; i; --i, ptr+=in.getStep())
                    *kptr++ = fac * *ptr;
        }
    }

    double* xdata = out.getData();
    fftw_complex* kdata = reinterpret_cast<fftw_complex*>(xdata);

    fftw_plan plan = fftw_plan_dft_c2r_2d(Ny, Nx, kdata, xdata, FFTW_ESTIMATE);
    if (plan==NULL) throw std::runtime_error("fftw_plan cannot be created");
    fftw_execute(plan);
    fftw_destroy_plan(plan);
}

template <typename T>
void cfft(const BaseImage<T>& in, ImageView<std::complex<double> > out,
          bool inverse, bool shift_in, bool shift_out)
{
    dbg<<"Start cfft\n";
    dbg<<"self bounds = "<<in.getBounds()<<std::endl;

    if (!in.getData() or !in.getBounds().isDefined())
        throw ImageError("Attempting to perform cfft on undefined image.");

    const int Nxo2 = in.getBounds().getXMax()+1;
    const int Nyo2 = in.getBounds().getYMax()+1;
    const int Nx = Nxo2 << 1;
    const int Ny = Nyo2 << 1;
    dbg<<"Nx,Ny = "<<Nx<<','<<Ny<<std::endl;

    if (in.getBounds().getYMin() != -Nyo2 && (in.getBounds().getXMin() != -Nxo2))
        throw ImageError("cfft requires bounds to be (-Nx/2, Nx/2-1, -Ny/2, Ny/2-1)");

    if (out.getBounds().getXMin() != -Nxo2 || out.getBounds().getXMax() != Nxo2-1 ||
        out.getBounds().getYMin() != -Nyo2 || out.getBounds().getYMax() != Nyo2-1)
        throw ImageError("cfft requires out.bounds to be (-Nx/2, Nx/2-1, -Ny/2, Ny/2-1)");

    if ((uintptr_t) out.getData() % 16 != 0)
        throw ImageError("cfft requires out.data to be 16 byte aligned");

    const T* ptr = in.getData();
    const int skip = in.getNSkip();
    std::complex<double>* kptr = out.getData();

    if (shift_out) {
        double fac = inverse ? 1./(Nx*Ny) : 1.;
        if (shift_in && (Nxo2 + Nyo2) % 2 == 1) fac = -fac;
        if (in.getStep() == 1) {
            for (int j=Ny; j; --j, ptr+=skip, fac=-fac)
                for (int i=Nx; i; --i, fac=-fac)
                    *kptr++ = fac * *ptr++;
        } else {
            for (int j=Ny; j; --j, ptr+=skip, fac=-fac)
                for (int i=Nx; i; --i, ptr+=in.getStep(), fac=-fac)
                    *kptr++ = fac * *ptr;
        }
    } else if (inverse) {
        double fac = 1./(Nx*Ny);
        if (in.getStep() == 1) {
            for (int j=Ny; j; --j, ptr+=skip)
                for (int i=Nx; i; --i)
                    *kptr++ = fac * *ptr++;
        } else {
            for (int j=Ny; j; --j, ptr+=skip)
                for (int i=Nx; i; --i, ptr+=in.getStep())
                    *kptr++ = fac * *ptr;
        }
    } else {
        if (in.getStep() == 1) {
            for (int j=Ny; j; --j, ptr+=skip)
                for (int i=Nx; i; --i)
                    *kptr++ = *ptr++;
        } else {
            for (int j=Ny; j; --j, ptr+=skip)
                for (int i=Nx; i; --i, ptr+=in.getStep())
                    *kptr++ = *ptr;
        }
    }

    fftw_complex* kdata = reinterpret_cast<fftw_complex*>(out.getData());

    fftw_plan plan = fftw_plan_dft_2d(Ny, Nx, kdata, kdata, inverse ? FFTW_BACKWARD : FFTW_FORWARD,
                                      FFTW_ESTIMATE);
    if (plan==NULL) throw std::runtime_error("fftw_plan cannot be created");
    fftw_execute(plan);
    fftw_destroy_plan(plan);

    if (shift_in) {
        kptr = out.getData();
        double fac = 1.;
        for (int j=Ny; j; --j, fac=-fac)
            for (int i=Nx; i; --i, fac=-fac)
                *kptr++ *= fac;
    }
}

template <typename T>
void invertImage(ImageView<T> im)
{ im.invertSelf(); }

// The classes ConstReturn, ReturnInverse, and ReturnSecond are defined in ImageArith.h.

// Helper function to set data to all zeros.  (memset doesn't like being called on complex<T>)
template <typename T>
void ZeroData(T* data, int n)
{ memset(data, 0, n * sizeof(T)); }

template <typename T>
void ZeroData(std::complex<T>* data, int n)
{ memset(reinterpret_cast<T*>(data), 0, 2 * n * sizeof(T)); }

template <typename T>
void ImageView<T>::fill(T x)
{
    if (x == T(0) && this->getNSkip()==0 && this->getStep()==1)
        ZeroData(this->getData(), this->getNElements());
    else
        transform_pixel(*this, ConstReturn<T>(x));
}

template <typename T>
void ImageView<T>::invertSelf()
{
    transform_pixel(*this, ReturnInverse<T>());
}

template <typename T>
void ImageView<T>::copyFrom(const BaseImage<T>& rhs)
{
    if (!this->_bounds.isSameShapeAs(rhs.getBounds()))
        throw ImageError("Attempt im1 = im2, but bounds not the same shape");
    transform_pixel(*this, rhs, ReturnSecond<T>());
}

// A helper function that will return the smallest 2^n or 3x2^n value that is
// even and >= the input integer.
int goodFFTSize(int input)
{
    if (input<=2) return 2;
    // Reduce slightly to eliminate potential rounding errors:
    double insize = (1.-1.e-5)*input;
    double log2n = std::log(2.)*std::ceil(std::log(insize)/std::log(2.));
    double log2n3 = std::log(3.)
        + std::log(2.)*std::ceil((std::log(insize)-std::log(3.))/std::log(2.));
    log2n3 = std::max(log2n3, std::log(6.)); // must be even number
    int Nk = int(std::ceil(std::exp(std::min(log2n, log2n3))-1.e-5));
    return Nk;
}

// Some Image arithmetic that can be sped up with SSE

// First the default implementation that we get if SSE isn't available.
template <typename T>
struct InnerLoopHelper
{
    template <typename T1, typename T2>
    static inline void mult_const(T1*& ptr, T2 x, int n)
    { for (; n; --n) *ptr++ *= x; }

    template <typename T1, typename T2>
    static inline void mult_im(T1*& p1, const T2*& p2, int n)
    { for (; n; --n) *p1++ *= *p2++; }
};

#ifdef __SSE__
template <>
struct InnerLoopHelper<float>
{
    static inline void mult_const(float*& ptr, float x, int n)
    {
        for (; n && !IsAligned(ptr); --n) *ptr++ *= x;

        int n4 = n>>2;
        int na = n4<<2;
        n -= na;

        if (n4) {
            __m128 xx = _mm_set1_ps(x);
            do {
                _mm_store_ps(ptr, _mm_mul_ps(_mm_load_ps(ptr), xx));
                ptr += 4;
            } while (--n4);
        }

        for (; n; --n) *ptr++ *= x;
    }
    static inline void mult_const(std::complex<float>*& ptr, float x, int n)
    {
        float* fptr = reinterpret_cast<float*>(ptr);
        mult_const(fptr, x, 2*n);
        ptr += n;
    }
    static inline void mult_const(std::complex<float>*& ptr, std::complex<float> x, int n)
    {
        for (; n && !IsAligned(ptr); --n) *ptr++ *= x;

        int n2 = n>>1;
        int na = n2<<1;
        n -= na;

        if (n2) {
            const float xr = x.real();
            const float xi = x.imag();
            __m128 xxr = _mm_set1_ps(xr);
            __m128 xxi = _mm_set_ps(xi, -xi, xi, -xi);
            do {
                __m128 p = _mm_load_ps(reinterpret_cast<float*>(ptr));
                __m128 pir = _mm_shuffle_ps(p, p, _MM_SHUFFLE(2,3,0,1));  // (pi, pr)
                __m128 z1 = _mm_mul_ps(xxr, p);    // (xr * pr, xr * pi)
                __m128 z2 = _mm_mul_ps(xxi, pir);  // (-xi * pi, xi * pr)
                __m128 z = _mm_add_ps(z1, z2);     // (xr pr - xi pi, xr pi + xi pr)
                _mm_store_ps(reinterpret_cast<float*>(ptr), z);
                ptr += 2;
            } while (--n2);
        }

        for (; n; --n) *ptr++ *= x;
    }

    static inline void mult_im(float*& p1, const float*& p2, int n)
    {
        for (; n && !IsAligned(p1); --n) *p1++ *= *p2++;

        int n4 = n>>2;
        int na = n4<<2;
        n -= na;

        for (; n4; --n4) {
            _mm_store_ps(p1, _mm_mul_ps(_mm_load_ps(p1), _mm_loadu_ps(p2)));
            p1 += 4;
            p2 += 4;
        }

        for (; n; --n) *p1++ *= *p2++;
    }
    static inline void mult_im(std::complex<float>*& p1, const float*& p2, int n)
    {
        for (; n && !IsAligned(p1); --n) *p1++ *= *p2++;

        int n4 = n>>2;
        int na = n4<<2;
        n -= na;

        for (; n4; --n4) {
            __m128 xp2 = _mm_loadu_ps(p2);                           // This holds 4 real from p2
            p2 += 4;
            __m128 xp1 = _mm_load_ps(reinterpret_cast<float*>(p1));  // This holds 2 complex from p1
            __m128 prod = _mm_mul_ps(xp1, _mm_unpacklo_ps(xp2,xp2)); // First 2 products
            _mm_store_ps(reinterpret_cast<float*>(p1), prod);
            p1 += 2;
            xp1 = _mm_load_ps(reinterpret_cast<float*>(p1));         // This is now the next 2.
            prod = _mm_mul_ps(xp1, _mm_unpackhi_ps(xp2,xp2));        // Next 2 products
            _mm_store_ps(reinterpret_cast<float*>(p1), prod);
            p1 += 2;
        }

        for (; n; --n) *p1++ *= *p2++;
    }
    static inline void mult_im(std::complex<float>*& p1, const std::complex<float>*& p2, int n)
    {
        for (; n && !IsAligned(p1); --n) *p1++ *= *p2++;

        int n2 = n>>1;
        int na = n2<<1;
        n -= na;

        if (n2) {
            const __m128 mneg = _mm_set_ps(1, -1, 1, -1);
            do {
                __m128 xp2 = _mm_loadu_ps(reinterpret_cast<const float*>(p2));
                p2 += 2;
                __m128 xp1 = _mm_load_ps(reinterpret_cast<float*>(p1));
                __m128 p1ir = _mm_shuffle_ps(xp1, xp1, _MM_SHUFFLE(2,3,0,1));
                __m128 p2rr = _mm_shuffle_ps(xp2, xp2, _MM_SHUFFLE(2,2,0,0));
                __m128 p2ii = _mm_shuffle_ps(xp2, xp2, _MM_SHUFFLE(3,3,1,1));
                p2ii = _mm_mul_ps(mneg, p2ii);
                __m128 z1 = _mm_mul_ps(p2rr, xp1);
                __m128 z2 = _mm_mul_ps(p2ii, p1ir);
                __m128 z = _mm_add_ps(z1, z2);
                _mm_store_ps(reinterpret_cast<float*>(p1), z);
                p1 += 2;
            } while (--n2);
        }

        if (n) *p1++ *= *p2++;
    }
};
#endif

#ifdef __SSE2__
template <>
struct InnerLoopHelper<double>
{
    static inline void mult_const(double*& ptr, double x, int n)
    {
        for (; n && !IsAligned(ptr); --n) *ptr++ *= x;

        int n2 = n>>1;
        int na = n2<<1;
        n -= na;

        if (n2) {
            __m128d xx = _mm_set1_pd(x);
            do {
                _mm_store_pd(ptr, _mm_mul_pd(_mm_load_pd(ptr), xx));
                ptr += 2;
            } while (--n2);
        }

        if (n) *ptr++ *= x;
    }
    static inline void mult_const(std::complex<double>*& ptr, double x, int n)
    {
        double* fptr = reinterpret_cast<double*>(ptr);
        mult_const(fptr, x, 2*n);
        ptr += n;
    }
    static inline void mult_const(std::complex<double>*& ptr, std::complex<double> x, int n)
    {
        for (; n && !IsAligned(ptr); --n) *ptr++ *= x;

        if (n) {
            const double xr = x.real();
            const double xi = x.imag();
            __m128d xxr = _mm_set1_pd(xr);
            __m128d xxi = _mm_set_pd(xi, -xi);
            do {
                __m128d p = _mm_load_pd(reinterpret_cast<double*>(ptr));
                __m128d pir = _mm_shuffle_pd(p, p, _MM_SHUFFLE2(0,1));  // (pi, pr)
                __m128d z1 = _mm_mul_pd(xxr, p);    // (xr * pr, xr * pi)
                __m128d z2 = _mm_mul_pd(xxi, pir);  // (-xi * pi, xi * pr)
                __m128d z = _mm_add_pd(z1, z2);     // (xr pr - xi pi, xr pi + xi pr)
                _mm_store_pd(reinterpret_cast<double*>(ptr++), z);
            } while (--n);
        }
    }

    static inline void mult_im(double*& p1, const double*& p2, int n)
    {
        for (; n && !IsAligned(p1); --n) *p1++ *= *p2++;

        int n2 = n>>1;
        int na = n2<<1;
        n -= na;

        for (; n2; --n2) {
            _mm_store_pd(p1, _mm_mul_pd(_mm_load_pd(p1), _mm_loadu_pd(p2)));
            p1 += 2;
            p2 += 2;
        }

        if (n) *p1++ *= *p2++;
    }
    static inline void mult_im(std::complex<double>*& p1, const double*& p2, int n)
    {
        for (; n && !IsAligned(p1); --n) *p1++ *= *p2++;

        int n2 = n>>1;
        int na = n2<<1;
        n -= na;

        for (; n2; --n2) {
            __m128d xp2 = _mm_loadu_pd(p2);
            p2 += 2;
            __m128d xp1 = _mm_load_pd(reinterpret_cast<double*>(p1));
            __m128d prod = _mm_mul_pd(xp1, _mm_unpacklo_pd(xp2,xp2));
            _mm_store_pd(reinterpret_cast<double*>(p1++), prod);
            xp1 = _mm_load_pd(reinterpret_cast<double*>(p1));
            prod = _mm_mul_pd(xp1, _mm_unpackhi_pd(xp2,xp2));
            _mm_store_pd(reinterpret_cast<double*>(p1++), prod);
        }

        if (n) *p1++ *= *p2++;
    }
    static inline void mult_im(std::complex<double>*& p1, const std::complex<double>*& p2, int n)
    {
        for (; n && !IsAligned(p1); --n) *p1++ *= *p2++;

        if (n) {
            const __m128d mneg = _mm_set_pd(1, -1);
            do {
                __m128d xp2 = _mm_loadu_pd(reinterpret_cast<const double*>(p2++));
                __m128d xp1 = _mm_load_pd(reinterpret_cast<double*>(p1));
                __m128d p1ir = _mm_shuffle_pd(xp1, xp1, _MM_SHUFFLE2(0,1));
                __m128d p2rr = _mm_shuffle_pd(xp2, xp2, _MM_SHUFFLE2(0,0));
                __m128d p2ii = _mm_shuffle_pd(xp2, xp2, _MM_SHUFFLE2(1,1));
                p2ii = _mm_mul_pd(mneg, p2ii);
                __m128d z1 = _mm_mul_pd(p2rr, xp1);
                __m128d z2 = _mm_mul_pd(p2ii, p1ir);
                __m128d z = _mm_add_pd(z1, z2);
                _mm_store_pd(reinterpret_cast<double*>(p1++), z);
            } while (--n);
        }
    }
};
#endif

template <typename T1, typename T2>
inline ImageView<T1>& MultConst(ImageView<T1>& im, T2 x)
{
    typedef typename ComplexHelper<T1>::real_type RT;
    T1* ptr = im.getData();
    if (ptr) {
        const int skip = im.getNSkip();
        const int step = im.getStep();
        const int nrow = im.getNRow();
        const int ncol = im.getNCol();
        if (step == 1) {
            for (int j=0; j<nrow; j++, ptr+=skip)
                InnerLoopHelper<RT>::mult_const(ptr, x, ncol);
        } else {
            for (int j=0; j<nrow; j++, ptr+=skip)
                for (int i=0; i<ncol; i++, ptr+=step) *ptr *= x;
        }
    }
    return im;
}

ImageView<float> operator*=(ImageView<float> im, float x)
{ return MultConst(im,x); }
ImageView<std::complex<float> > operator*=(ImageView<std::complex<float> > im, float x)
{ return MultConst(im,x); }
ImageView<std::complex<float> > operator*=(ImageView<std::complex<float> > im,
                                           std::complex<float> x)
{ return MultConst(im,x); }

ImageView<double> operator*=(ImageView<double> im, double x)
{ return MultConst(im,x); }
ImageView<std::complex<double> > operator*=(ImageView<std::complex<double> > im, double x)
{ return MultConst(im,x); }
ImageView<std::complex<double> > operator*=(ImageView<std::complex<double> > im,
                                            std::complex<double> x)
{ return MultConst(im,x); }

template <typename T1, typename T2>
inline ImageView<T1>& MultIm(ImageView<T1>& im1, const BaseImage<T2>& im2)
{
    typedef typename ComplexHelper<T1>::real_type RT;
    T1* ptr1 = im1.getData();
    if (ptr1) {
        const int skip1 = im1.getNSkip();
        const int step1 = im1.getStep();
        const int nrow = im1.getNRow();
        const int ncol = im1.getNCol();
        const T2* ptr2 = im2.getData();
        const int skip2 = im2.getNSkip();
        const int step2 = im2.getStep();
        if (step1 == 1 && step2 == 1) {
            for (int j=0; j<nrow; j++, ptr1+=skip1, ptr2+=skip2)
                InnerLoopHelper<RT>::mult_im(ptr1, ptr2, ncol);
        } else {
            for (int j=0; j<nrow; j++, ptr1+=skip1, ptr2+=skip2)
                for (int i=0; i<ncol; i++, ptr1+=step1, ptr2+=step2) *ptr1 *= *ptr2;
        }
    }
    return im1;
}

ImageView<float> operator*=(ImageView<float> im1, const BaseImage<float>& im2)
{ return MultIm(im1,im2); }
ImageView<std::complex<float> > operator*=(ImageView<std::complex<float> > im1,
                                           const BaseImage<float>& im2)
{ return MultIm(im1,im2); }
ImageView<std::complex<float> > operator*=(ImageView<std::complex<float> > im1,
                                           const BaseImage<std::complex<float> >& im2)
{ return MultIm(im1,im2); }

ImageView<double> operator*=(ImageView<double> im1, const BaseImage<double>& im2)
{ return MultIm(im1,im2); }
ImageView<std::complex<double> > operator*=(ImageView<std::complex<double> > im1,
                                            const BaseImage<double>& im2)
{ return MultIm(im1,im2); }
ImageView<std::complex<double> > operator*=(ImageView<std::complex<double> > im1,
                                            const BaseImage<std::complex<double> >& im2)
{ return MultIm(im1,im2); }



// instantiate for expected types

#define T double
#include "Image.inst"
#undef T

#define T float
#include "Image.inst"
#undef T

#define T int32_t
#include "Image.inst"
#undef T

#define T int16_t
#include "Image.inst"
#undef T

#define T uint32_t
#include "Image.inst"
#undef T

#define T uint16_t
#include "Image.inst"
#undef T

#define T std::complex<double>
#include "Image.inst"
#undef T

#define T std::complex<float>
#include "Image.inst"
#undef T


} // namespace galsim

