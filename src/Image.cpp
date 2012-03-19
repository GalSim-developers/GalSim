#include "Image.h"
#include <sstream>

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
ImageBounds::ImageBounds(
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
        oss << "Attempt to access row number "<<x
            << ", range is "<<b.getYMin()<<" to "<<b.getYMax();
        found = true;
    } 
    if (!found) return "Cannot find bounds violation ???";
    else return oss.str();
}
ImageBounds::ImageBounds(const int x, const int y, const Bounds<int> b) :
    ImageError(MakeErrorMessage(x,y,b)) 
{}


/////////////////////////////////////////////////////////////////////
//// Image<const T> Implementation
///////////////////////////////////////////////////////////////////////

namespace {

template <typename T>
class ArrayDeleter {
public:
    void operator()(T * p) const { delete [] p; }
};

} // anonymous

template <typename T>
Image<const T>::Image(const int ncol, const int nrow) :
    _owner(), _data(0), _stride(ncol), _scale(1.0), _bounds(1, ncol, 1, nrow)
{
    int nElements = ncol * nrow;
    if (nElements) {
        _owner.reset(new T[nElements], ArrayDeleter<T>());
        _data = _owner.get();
        std::fill(_data, _data + nElements, T(0));
    }
}

template <typename T>
Image<const T>::Image(const Bounds<int> & bounds, const T initValue) :
    _owner(), _data(0), _stride(0), _scale(1.0), _bounds(bounds)
{
    if (_bounds.isDefined()) {
        _stride = _bounds.getXMax() - _bounds.getXMin() + 1;
        int nElements = _stride * (_bounds.getYMax() - _bounds.getYMin() + 1);
        _owner.reset(new T[nElements], ArrayDeleter<T>());
        _data = _owner.get();
        std::fill(_data, _data + nElements, initValue);
    }
}

template <typename T>
Image<const T>::Image(
    const T * data,
    boost::shared_ptr<const T> const & owner, 
    int stride,
    const Bounds<int> & bounds
) : _owner(boost::const_pointer_cast<T>(owner)), _data(const_cast<T*>(data)), _stride(stride),
    _scale(1.0), _bounds(bounds)
{}

template <typename T>
void Image<const T>::redefine(const Bounds<int> & bounds) {
    _data += (bounds.getYMin() - _bounds.getYMin()) * _stride
        + (bounds.getXMin() - _bounds.getXMin());
    _bounds = bounds;
}

template <typename T>
Image<T> Image<const T>::duplicate() const {
    Image<T> result(_bounds);
    result.copyFrom(*this);
    return result;
}

template <typename T>
Image<const T> Image<const T>::subimage(const Bounds<int> & bounds) const {
    if (!_bounds.includes(bounds)) {
        std::ostringstream os;
        os << "Subimage bounds (" << bounds << ") are outside original image bounds (" 
           << _bounds << ")";
        throw ImageError(os.str());
    }
    Image<const T> result(*this);
    result.redefine(bounds);
    return result;
}

template <typename T>
void Image<const T>::resize(const Bounds<int> & bounds) {
    if ((bounds.getXMax() - bounds.getXMin()) == (_bounds.getXMax() - _bounds.getXMin())
        && (bounds.getYMax() - bounds.getYMin()) == (_bounds.getYMax() - _bounds.getYMin())) {
        _bounds = bounds;
    } else {
        *this = Image<const T>(bounds);
    }
}

// note: There are lots of unnecessary bbox intersections in these, but I'll keep the
// implementation simple unless we know we need to optimize it.

template <typename T>
Image<T> Image<const T>::operator+(const Image<const T> & rhs) const {
    Image<T> result(this->_bounds & rhs.getBounds());
    result.copyFrom(*this);
    result += rhs;
    return result;
}

template <typename T>
Image<T> Image<const T>::operator-(const Image<const T> & rhs) const {
    Image<T> result(this->_bounds & rhs.getBounds());
    result.copyFrom(*this);
    result -= rhs;
    return result;
}

template <typename T>
Image<T> Image<const T>::operator*(const Image<const T> & rhs) const {
    Image<T> result(this->_bounds & rhs.getBounds());
    result.copyFrom(*this);
    result *= rhs;
    return result;
}

template <typename T>
Image<T> Image<const T>::operator/(const Image<const T> & rhs) const {
    Image<T> result(this->_bounds & rhs.getBounds());
    result.copyFrom(*this);
    result /= rhs;
    return result;
}

/////////////////////////////////////////////////////////////////////
//// Image<T> Implementation
///////////////////////////////////////////////////////////////////////

template <typename T>
Image<T> Image<T>::subimage(const Bounds<int> & bounds) const {
    if (!this->_bounds.includes(bounds)) {
        std::ostringstream os;
        os << "Subimage bounds (" << bounds << ") are outside original image bounds (" 
           << this->_bounds << ")";
        throw ImageError(os.str());
    }
    Image<T> result(*this);
    result.redefine(bounds);
    return result;
}

namespace {

template <typename T>
class ConstReturn {
public: 
    ConstReturn(const T v): val(v) {}
    T operator()(const T dummy) const { return val; }
private:
    T val;
};

template <typename T>
class ReturnSecond {
public:
    T operator()(T, T v) const { return v; }
};

} // anonymous

template <typename T>
void Image<T>::fill(T x) const {
    transform_pixel(*this, ConstReturn<T>(x));
}

template <typename T>
Image<T> const & Image<T>::operator+=(T x) const {
    transform_pixel(*this, bind2nd(std::plus<T>(),x));
    return *this;
}

template <typename T>
Image<T> const & Image<T>::operator-=(T x) const {
    transform_pixel(*this, bind2nd(std::minus<T>(),x));
    return *this;
}

template <typename T>
Image<T> const & Image<T>::operator*=(T x) const {
    transform_pixel(*this, bind2nd(std::multiplies<T>(),x));
    return *this;
}

template <typename T>
Image<T> const & Image<T>::operator/=(T x) const {
    transform_pixel(*this, bind2nd(std::divides<T>(),x));
    return *this;
}

template <typename T>
void Image<T>::copyFrom(const Image<const T> & rhs) {
    transform_pixel(*this, rhs, this->_bounds & rhs.getBounds(), ReturnSecond<T>());
}

template <typename T>
Image<T> const & Image<T>::operator+=(const Image<const T> & rhs) const {
    transform_pixel(*this, rhs, this->_bounds & rhs.getBounds(), std::plus<T>());
    return *this;
}

template <typename T>
Image<T> const & Image<T>::operator-=(const Image<const T> & rhs) const {
    transform_pixel(*this, rhs, this->_bounds & rhs.getBounds(), std::minus<T>());
    return *this;    
}

template <typename T>
Image<T> const & Image<T>::operator*=(const Image<const T> & rhs) const {
    transform_pixel(*this, rhs, this->_bounds & rhs.getBounds(), std::multiplies<T>());
    return *this;    
}

template <typename T>
Image<T> const & Image<T>::operator/=(const Image<const T> & rhs) const {
    transform_pixel(*this, rhs, this->_bounds & rhs.getBounds(), std::divides<T>());
    return *this;    
}

// instantiate for expected types

template class Image<double>;
template class Image<float>;
template class Image<int>;
template class Image<short>;
template class Image<const double>;
template class Image<const float>;
template class Image<const int>;
template class Image<const short>;

} // namespace galsim

