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

#include <sstream>

#include "Image.h"
#include "ImageArith.h"
#include "FFT.h"
#include "SBInterpolatedImage.h"

namespace galsim {

/////////////////////////////////////////////////////////////////////
//// Constructor for out-of-bounds that has coordinate info
///////////////////////////////////////////////////////////////////////


std::string MakeErrorMessage(
    const std::string& m, const int min, const int max, const int tried)
{
    std::ostringstream oss;
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
    std::ostringstream oss;
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
BaseImage<T>::BaseImage(const Bounds<int>& b, double scale) :
    AssignableToImage<T>(b), _owner(), _data(0), _stride(0), _scale(scale)
{
    if (this->_bounds.isDefined()) allocateMem();
    // Else _data is left as 0, stride = 0.
}

template <typename T>
void BaseImage<T>::allocateMem()
{
    // Note: this version always does the memory (re-)allocation. 
    // So the various functions that call this should do their (different) checks 
    // for whether this is necessary.
    _stride = this->_bounds.getXMax() - this->_bounds.getXMin() + 1;

    ptrdiff_t nElements = _stride * (this->_bounds.getYMax() - this->_bounds.getYMin() + 1);
    if (_stride <= 0 || nElements <= 0) {
        std::ostringstream oss;
        oss << "Attempt to create an Image with defined but invalid Bounds ("<<this->_bounds<<")\n";
        throw ImageError(oss.str());
    }

    // The ArrayDeleter is because we use "new T[]" rather than an normal new.
    // Without ArrayDeleter, shared_ptr would just use a regular delete, rather
    // than the required "delete []".
    _owner.reset(new T[nElements], ArrayDeleter<T>());
    _data = _owner.get();
}

template <typename T>
Image<T>::Image(int ncol, int nrow, T init_value) : BaseImage<T>(Bounds<int>(1,ncol,1,nrow)) 
{
    if (ncol <= 0 || nrow <= 0) {
        std::ostringstream oss;
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
Image<T>::Image(const Bounds<int>& bounds, const T init_value) : BaseImage<T>(bounds)
{
    fill(init_value);
}

template <typename T>
void Image<T>::resize(const Bounds<int>& new_bounds) 
{
    if (!new_bounds.isDefined()) {
        // Then this is really a deallocation.  Clear out the existing memory.
        this->_owner.reset();
        this->_data = 0;
        this->_stride = 0;
    } else if (this->_bounds.isDefined() && 
               this->_bounds.area() <= new_bounds.area() && 
               this->_owner.unique()) {
        // Then safe to keep existing memory allocation.
        // Just redefine the bounds and stride.
        this->_stride = new_bounds.getXMax() - new_bounds.getXMin() + 1;
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
T& ImageView<T>::at(const int xpos, const int ypos) const
{
    if (!this->_data) throw ImageError("Attempt to access values of an undefined image");
    if (!this->_bounds.includes(xpos, ypos)) throw ImageBoundsError(xpos, ypos, this->_bounds);
    return this->_data[this->addressPixel(xpos, ypos)];
}

template <typename T>
T& Image<T>::at(const int xpos, const int ypos)
{
    if (!this->_data) throw ImageError("Attempt to access values of an undefined image");
    if (!this->_bounds.includes(xpos, ypos)) throw ImageBoundsError(xpos, ypos, this->_bounds);
    return this->_data[this->addressPixel(xpos, ypos)];
}

template <typename T>
const T& Image<T>::at(const int xpos, const int ypos) const
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
        std::ostringstream os;
        os << "Subimage bounds (" << bounds << ") are outside original image bounds (" 
           << this->_bounds << ")";
        throw ImageError(os.str());
    }
    T* newdata = _data
        + (bounds.getYMin() - this->_bounds.getYMin()) * _stride
        + (bounds.getXMin() - this->_bounds.getXMin());
    return ConstImageView<T>(newdata,_owner,_stride,bounds,this->_scale);
}

template <typename T>
ImageView<T> ImageView<T>::subImage(const Bounds<int>& bounds) const 
{
    if (!this->_data) throw ImageError("Attempt to make subImage of an undefined image");
    if (!this->_bounds.includes(bounds)) {
        std::ostringstream os;
        os << "Subimage bounds (" << bounds << ") are outside original image bounds (" 
           << this->_bounds << ")";
        throw ImageError(os.str());
    }
    T* newdata = this->_data
        + (bounds.getYMin() - this->_bounds.getYMin()) * this->_stride
        + (bounds.getXMin() - this->_bounds.getXMin());
    return ImageView<T>(newdata,this->_owner,this->_stride,bounds,this->_scale);
}

template <typename T>
int BaseImage<T>::getPaddedSize(float pad_factor) const
{
    int Ninitial = std::max(this->getYMax()-this->getYMin()+1,
                            this->getXMax()-this->getXMin()+1 );
    Ninitial = Ninitial + Ninitial%2;
    assert(Ninitial%2==0);
    assert(Ninitial>=2);

    if (pad_factor <= 0.) pad_factor = sbp::default_pad_factor;
    return goodFFTSize(int(pad_factor*Ninitial));
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
    T operator()(const T val) const { return val==T(0) ? T(0) : T(1)/val; }
};

template <typename T>
class ReturnSecond 
{
public:
    T operator()(T, T v) const { return v; }
};

} // anonymous

template <typename T>
void ImageView<T>::fill(T x) const 
{
    transform_pixel(*this, ConstReturn<T>(x));
}

template <typename T>
void ImageView<T>::invertSelf() const 
{
    transform_pixel(*this, ReturnInverse<T>());
}

template <typename T>
void ImageView<T>::copyFrom(const BaseImage<T>& rhs) const
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
template class Image<double>;
template class Image<float>;
template class Image<int32_t>;
template class Image<int16_t>;
template class ImageView<double>;
template class ImageView<float>;
template class ImageView<int32_t>;
template class ImageView<int16_t>;
template class ConstImageView<double>;
template class ConstImageView<float>;
template class ConstImageView<int32_t>;
template class ConstImageView<int16_t>;

} // namespace galsim

