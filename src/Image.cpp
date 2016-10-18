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

#include "Image.h"
#include "ImageArith.h"
#include "FFT.h"

namespace galsim {

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

namespace {

template <typename T>
class ArrayDeleter {
public:
    void operator()(T * p) const { delete [] p; }
};

} // anonymous

template <typename T>
BaseImage<T>::BaseImage(const Bounds<int>& b) :
    AssignableToImage<T>(b), _owner(), _data(0), _nElements(0), _step(0), _stride(0),
    _ncol(0), _nrow(0)
{
    if (this->_bounds.isDefined()) allocateMem();
    // Else _data is left as 0, step,stride = 0.
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

    // The ArrayDeleter is because we use "new T[]" rather than an normal new.
    // Without ArrayDeleter, shared_ptr would just use a regular delete, rather
    // than the required "delete []".
    _owner.reset(new T[_nElements], ArrayDeleter<T>());
    _data = _owner.get();
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

// A helper function so we can write CONJ(x) for real or complex.
template <typename T>
struct ConjHelper
{ static T conj(const T& x) { return x; } };
template <typename T>
struct ConjHelper<std::complex<T> >
{ static std::complex<T> conj(const std::complex<T>& x) { return std::conj(x); } };
template <typename T>
T CONJ(const T& x) { return ConjHelper<T>::conj(x); }


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

    dbg<<"Start hermx_cols_pair\n";
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
    dbg<<"Start hermx_cols\n";
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
        T* ptrwrap;
        for (int j=j1; j<j2; ++j, ptr+=skip) {
            xdbg<<"Wrap row "<<j<<" into columns ["<<i1<<','<<i2<<")\n";
            xdbg<<"ptr = "<<ptr-this->getData()<<std::endl;
            wrap_cols(ptr, m, mwrap, i1, i2, step);
        }
    }

    return ret;
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

// instantiate for expected types

template class BaseImage<double>;
template class BaseImage<float>;
template class BaseImage<int32_t>;
template class BaseImage<int16_t>;
template class BaseImage<std::complex<double> >;
template class ImageAlloc<double>;
template class ImageAlloc<float>;
template class ImageAlloc<int32_t>;
template class ImageAlloc<int16_t>;
template class ImageAlloc<std::complex<double> >;
template class ImageView<double>;
template class ImageView<float>;
template class ImageView<int32_t>;
template class ImageView<int16_t>;
template class ImageView<std::complex<double> >;
template class ConstImageView<double>;
template class ConstImageView<float>;
template class ConstImageView<int32_t>;
template class ConstImageView<int16_t>;
template class ConstImageView<std::complex<double> >;

} // namespace galsim

