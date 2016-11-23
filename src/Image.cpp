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

//#define DEBUGLOGGING

#include <sstream>
#include <numeric>

#include "fftw3.h"

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

    // This bit is based on the answers here:
    // http://stackoverflow.com/questions/227897/how-to-allocate-aligned-memory-only-using-the-standard-library/227900
    // The point of this is to get the _data pointer aligned to a 16 byte (128 bit) boundary.
    // Arrays that are so aligned can use SSE operations and so can be much faster than
    // non-aligned memroy.  FFTW in particular is faster if it gets aligned data.
    char* mem = new char[_nElements * sizeof(T) + sizeof(char*) + 15];
    _data = reinterpret_cast<T*>( (uintptr_t)(mem + sizeof(char*) + 15) & ~(size_t) 0x0F );
    ((char**)_data)[-1] = mem;
    _owner.reset(_data, AlignedDeleter<T>());
}

template <typename T>
struct Sum
{
    Sum(): sum(0) {}
    void operator()(T x) { sum += x; }
    T sum;
};

template <typename T>
T BaseImage<T>::sumElements() const
{
    Sum<T> sum;
    sum = for_each_pixel(*this, sum);
    return sum.sum;
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
    nz = for_each_pixel_ij(*this, nz);
    return nz.bounds;
}

template <typename T>
ImageAlloc<T>::ImageAlloc(int ncol, int nrow, T init_value) :
    BaseImage<T>(Bounds<int>(1,ncol,1,nrow))
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
    fill(init_value);
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
        for(int i=0; i<m; ++i) *ptrwrap++ += *ptr++;
    else
        for(int i=0; i<m; ++i,ptr+=step,ptrwrap+=step) *ptrwrap += *ptr;
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
        for(int i=0; i<m; ++i) *ptrwrap-- += CONJ(*ptr++);
    else
        for(int i=0; i<m; ++i,ptr+=step,ptrwrap-=step) *ptrwrap += CONJ(*ptr);
}

// If j == jj, this needs to be slightly different.
template <typename T>
void wrap_row_selfconj(T*& ptr, T*& ptrwrap, int m, int step)
{
    if (step == 1)
        for(int i=0; i<(m+1)/2; ++i,++ptr,--ptrwrap) {
            *ptrwrap += CONJ(*ptr);
            *ptr = CONJ(*ptrwrap);
        }
    else
        for(int i=0; i<(m+1)/2; ++i,ptr+=step,ptrwrap-=step) {
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
ImageView<T> ImageView<T>::wrap(const Bounds<int>& b, bool hermx, bool hermy)
{
    // Get this at the start to check for invalid bounds and raise the exception before
    // possibly writing data past the edge of the image.
    ImageView<T> ret = subImage(b);

    dbg<<"Start ImageView::wrap: b = "<<b<<std::endl;
    dbg<<"self bounds = "<<this->_bounds<<std::endl;
    //set_verbose(2);

    const int i1 = b.getXMin()-this->_bounds.getXMin();
    const int i2 = b.getXMax()-this->_bounds.getXMin()+1;  // +1 for "1 past the end"
    const int j1 = b.getYMin()-this->_bounds.getYMin();
    const int j2 = b.getYMax()-this->_bounds.getYMin()+1;
    xdbg<<"i1,i2,j1,j2 = "<<i1<<','<<i2<<','<<j1<<','<<j2<<std::endl;
    const int mwrap = i2-i1;
    const int nwrap = j2-j1;
    const int skip = this->getNSkip();
    const int step = this->getStep();
    const int stride = this->getStride();
    const int m = this->getNCol();
    const int n = this->getNRow();
    T* ptr = this->getData();

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

        T* ptr1 = getData() + (i2-1)*step;
        T* ptr2 = getData() + (n-1)*stride + (i2-1)*step;

        // These skips will take us from the end of one row to the i2-1 element in the next row.
        int skip1 = skip + (i2-1)*step;
        int skip2 = skip1 - 2*stride; // This is negative.  We add this value to ptr2.

        for (int j=0; j<mid; ++j, ptr1+=skip1, ptr2+=skip2) {
            xdbg<<"Wrap rows "<<j<<","<<n-j-1<<" into columns ["<<i1<<','<<i2<<")\n";
            xdbg<<"ptrs = "<<ptr1-this->getData()<<"  "<<ptr2-this->getData()<<std::endl;
            wrap_hermx_cols_pair(ptr1, ptr2, m, mwrap, step);
        }
        // Finally, the row that is really j=0 (but here is j=(n-1)/2) also needs to be wrapped
        // singly.
        xdbg<<"Wrap row "<<mid<<" into columns ["<<i1<<','<<i2<<")\n";
        xdbg<<"ptrs = "<<ptr1-this->getData()<<"  "<<ptr2-this->getData()<<std::endl;
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
        xdbg<<"ptrs = "<<ptr-this->getData()<<"  "<<ptrwrap-this->getData()<<std::endl;
        wrap_row_selfconj(ptr, ptrwrap, m, step);

        ptr += skip;
        ptrwrap -= skip;
        --jj;
        int  j= j2;
        while (1) {
            int k = std::min(n-j,jj);  // How many conjugate rows to do?
            for (; k; --k, ++j, --jj, ptr+=skip, ptrwrap-=skip) {
                xdbg<<"Wrap row "<<j<<" onto row = "<<jj<<" using conjugation.\n";
                xdbg<<"ptrs = "<<ptr-this->getData()<<"  "<<ptrwrap-this->getData()<<std::endl;
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
                xdbg<<"ptrs = "<<ptr-this->getData()<<"  "<<ptrwrap-this->getData()<<std::endl;
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
                xdbg<<"ptrs = "<<ptr-this->getData()<<"  "<<ptrwrap-this->getData()<<std::endl;
                wrap_row(ptr, ptrwrap, m, step);
            }
            jj = j1;
            ptrwrap -= nwrap * stride;
        }
    }

    // In the normal (not hermx) case, we now wrap rows [j1,j2) into the columns [i1,i2).
    if (!hermx) {
        ptr = getData() + j1*stride;
        for (int j=j1; j<j2; ++j, ptr+=skip) {
            xdbg<<"Wrap row "<<j<<" into columns ["<<i1<<','<<i2<<")\n";
            xdbg<<"ptr = "<<ptr-this->getData()<<std::endl;
            wrap_cols(ptr, m, mwrap, i1, i2, step);
        }
    }

    return ret;
}


template <typename T>
ImageView<std::complex<double> > BaseImage<T>::fft(bool shift_in, bool shift_out) const
{
    dbg<<"Start BaseImage::fft\n";
    dbg<<"self bounds = "<<this->_bounds<<std::endl;

    if (!_data or !this->_bounds.isDefined())
        throw ImageError("Attempting to perform fft on undefined image.");

    const int No2 = this->_bounds.getXMax()+1;
    const int N = No2 << 1;
    dbg<<"N = "<<N<<std::endl;

    if (this->_bounds.getYMin() != -No2 || this->_bounds.getYMax() != No2-1 ||
        this->_bounds.getXMin() != -No2)
        throw ImageError("fft requires bounds to be (-N/2, N/2-1, -N/2, N/2-1)");

    // ImageAlloc's memory allocation is aligned on 16 byte boundaries, which means we can
    // use it for the fftw array.
    // We will use the same array for input and output.  It's simplest if we create the
    // output image and just cast to double for the input.
    // However, note that the complex array has two extra elements in the primary direction
    // (x in our case) to allow for the extra column.
    // cf. http://www.fftw.org/doc/Real_002ddata-DFT-Array-Format.html
    ImageAlloc<std::complex<double> > kim(Bounds<int>(0, No2, -No2, No2-1));
    double* xptr = reinterpret_cast<double*>(kim.getData());
    const T* ptr = _data;
    const int skip = this->getNSkip();

    // The FT image that FFTW will return will have FT(0,0) placed at the origin.  We
    // want it placed in the middle instead.  We can make that happen by inverting every other
    // row in the input image.
    if (shift_out) {
        double fac = (shift_in && No2 % 2 == 1) ? -1 : 1.;
        if (_step == 1) {
            for (int j=-No2; j<No2; ++j, ptr+=skip, xptr+=2, fac=-fac)
                for (int i=-No2; i<No2; ++i)
                    *xptr++ = fac * REAL(*ptr++);
        } else {
            for (int j=-No2; j<No2; ++j, ptr+=skip, xptr+=2, fac=-fac)
                for (int i=-No2; i<No2; ++i, ptr+=_step)
                    *xptr++ = fac * REAL(*ptr);
        }
    } else {
        if (_step == 1) {
            for (int j=-No2; j<No2; ++j, ptr+=skip, xptr+=2)
                for (int i=-No2; i<No2; ++i)
                    *xptr++ = REAL(*ptr++);
        } else {
            for (int j=-No2; j<No2; ++j, ptr+=skip, xptr+=2)
                for (int i=-No2; i<No2; ++i, ptr+=_step)
                    *xptr++ = REAL(*ptr);
        }
    }

    fftw_complex* kdata = reinterpret_cast<fftw_complex*>(kim.getData());
    double* xdata = reinterpret_cast<double*>(kim.getData());

    fftw_plan plan = fftw_plan_dft_r2c_2d(N, N, xdata, kdata, FFTW_ESTIMATE);
    if (plan==NULL) throw std::runtime_error("fftw_plan cannot be created");
    fftw_execute(plan);
    fftw_destroy_plan(plan);

    // The resulting image will still have a checkerboard pattern of +-1 on it, which
    // we want to remove.
    if (shift_in) {
        std::complex<double>* kptr = kim.getData();
        double fac = 1.;
        const bool extra_flip = (No2 % 2 == 1);
        for (int j=-No2; j<No2; ++j, fac=(extra_flip?-fac:fac))
            for (int i=0; i<=No2; ++i, fac=-fac)
                *kptr++ *= fac;
    }

    // Now simply return a view of this image.
    return kim.view();
}

template <typename T>
ImageView<double> BaseImage<T>::inverse_fft(bool shift_in, bool shift_out) const
{
    dbg<<"Start BaseImage::inverse_fft\n";
    dbg<<"self bounds = "<<this->_bounds<<std::endl;

    if (!_data or !this->_bounds.isDefined())
        throw ImageError("Attempting to perform inverse fft on undefined image.");

    if (this->_bounds.getXMin() != 0)
        throw ImageError("inverse_fft requires bounds to be (0, N/2, -N/2, N/2-1)");

    const int No2 = this->_bounds.getXMax();
    const int N = No2 << 1;
    dbg<<"N = "<<N<<std::endl;

    if (this->_bounds.getYMin() != -No2 || this->_bounds.getYMax() != No2-1)
        throw ImageError("inverse_fft requires bounds to be (0, N/2, -N/2, N/2-1)");

    // ImageAlloc's memory allocation is aligned on 16 byte boundaries, which means we can
    // use it for the fftw array.
    // We will use the same array for input and output.  It's simplest if we create the
    // output image and just cast to complex for the input.
    // However, note that the real array needs two extra elements in the primary direction
    // (x in our case) to allow for the extra column in the k array.
    // cf. http://www.fftw.org/doc/Real_002ddata-DFT-Array-Format.html
    // The bounds we care about are (-No2, No2-1, -No2, No2-1).
    ImageAlloc<double> xim(Bounds<int>(-No2, No2+1, -No2, No2-1));

    std::complex<double>* kptr = reinterpret_cast<std::complex<double>*>(xim.getData());

    // FFTW wants the locations of the + and - ky values swapped relative to how
    // we store it in an image.
    // Also, to put x=0 in center of array, we need to flop the sign of every other element
    // and need to scale by (1/N)^2.
    double fac = 1./(N*N);

    const int start_offset = shift_in ? No2 * _stride : 0;
    const int mid_offset = shift_in ? 0 : No2 * _stride;

    const int skip = this->getNSkip();
    if (shift_out) {
        const T* ptr = _data + start_offset;
        const bool extra_flip = (No2 % 2 == 1);
        if (_step == 1) {
            for (int j=0; j<No2; ++j, ptr+=skip, fac=(extra_flip?-fac:fac))
                for (int i=0; i<=No2; ++i, fac=-fac)
                    *kptr++ = fac * *ptr++;
            ptr = _data + mid_offset;
            for (int j=-No2; j<0; ++j, ptr+=skip, fac=(extra_flip?-fac:fac))
                for (int i=0; i<=No2; ++i, fac=-fac)
                    *kptr++ = fac * *ptr++;
        } else {
            for (int j=0; j<No2; ++j, ptr+=skip, fac=(extra_flip?-fac:fac))
                for (int i=0; i<=No2; ++i, ptr+=_step, fac=-fac)
                    *kptr++ = fac * *ptr;
            ptr = _data + mid_offset;
            for (int j=No2; j<N; ++j, ptr+=skip, fac=(extra_flip?-fac:fac))
                for (int i=0; i<=No2; ++i, ptr+=_step, fac=-fac)
                    *kptr++ = fac * *ptr;
        }
    } else {
        const T* ptr = _data + start_offset;
        if (_step == 1) {
            for (int j=0; j<No2; ++j, ptr+=skip)
                for (int i=0; i<=No2; ++i)
                    *kptr++ = fac * *ptr++;
            ptr = _data + mid_offset;
            for (int j=-No2; j<0; ++j, ptr+=skip)
                for (int i=0; i<=No2; ++i)
                    *kptr++ = fac * *ptr++;
        } else {
            for (int j=0; j<No2; ++j, ptr+=skip)
                for (int i=0; i<=No2; ++i, ptr+=_step)
                    *kptr++ = fac * *ptr;
            ptr = _data + mid_offset;
            for (int j=No2; j<N; ++j, ptr+=skip)
                for (int i=0; i<=No2; ++i, ptr+=_step)
                    *kptr++ = fac * *ptr;
        }
    }

    double* xdata = xim.getData();
    fftw_complex* kdata = reinterpret_cast<fftw_complex*>(xdata);

    fftw_plan plan = fftw_plan_dft_c2r_2d(N, N, kdata, xdata, FFTW_ESTIMATE);
    if (plan==NULL) throw std::runtime_error("fftw_plan cannot be created");
    fftw_execute(plan);
    fftw_destroy_plan(plan);

    // Now simply return a view of this image.
    return xim.subImage(Bounds<int>(-No2, No2-1, -No2, No2-1));
}

namespace {

template <typename T>
class ConstReturn
{
public:
    ConstReturn(const T v): val(v) {}
    T operator()(const T ) const { return val; }
private:
    T val;
};

template <typename T>
class ReturnInverse
{
public:
    T operator()(const T val) const { return val==T(0) ? T(0.) : T(1./val); }
};

template <typename T>
class ReturnSecond
{
public:
    T operator()(T, T v) const { return v; }
};

} // anonymous

template <typename T>
void ImageView<T>::fill(T x)
{
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


// instantiate for expected types

template class BaseImage<double>;
template class BaseImage<float>;
template class BaseImage<int32_t>;
template class BaseImage<int16_t>;
template class BaseImage<uint32_t>;
template class BaseImage<uint16_t>;
template class BaseImage<std::complex<double> >;
template class ImageAlloc<double>;
template class ImageAlloc<float>;
template class ImageAlloc<int32_t>;
template class ImageAlloc<int16_t>;
template class ImageAlloc<uint32_t>;
template class ImageAlloc<uint16_t>;
template class ImageAlloc<std::complex<double> >;
template class ImageView<double>;
template class ImageView<float>;
template class ImageView<int32_t>;
template class ImageView<int16_t>;
template class ImageView<uint32_t>;
template class ImageView<uint16_t>;
template class ImageView<std::complex<double> >;
template class ConstImageView<double>;
template class ConstImageView<float>;
template class ConstImageView<int32_t>;
template class ConstImageView<int16_t>;
template class ConstImageView<uint32_t>;
template class ConstImageView<uint16_t>;
template class ConstImageView<std::complex<double> >;
} // namespace galsim

