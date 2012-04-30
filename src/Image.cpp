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
        oss << "Attempt to access row number "<<x
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
    AssignableToImage<T>(b), _owner(), _data(0), _stride(0)
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
    int nElements = _stride * (this->_bounds.getYMax() - this->_bounds.getYMin() + 1);
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
    if (!this->_bounds.includes(xpos, ypos)) throw ImageBoundsError(xpos, ypos, this->_bounds);
    return _data[addressPixel(xpos, ypos)];
}

template <typename T>
T& ImageView<T>::at(const int xpos, const int ypos) const
{
    if (!this->_bounds.includes(xpos, ypos)) throw ImageBoundsError(xpos, ypos, this->_bounds);
    return this->_data[this->addressPixel(xpos, ypos)];
}

template <typename T>
T& Image<T>::at(const int xpos, const int ypos)
{
    if (!this->_bounds.includes(xpos, ypos)) throw ImageBoundsError(xpos, ypos, this->_bounds);
    return this->_data[this->addressPixel(xpos, ypos)];
}

template <typename T>
const T& Image<T>::at(const int xpos, const int ypos) const
{
    if (!this->_bounds.includes(xpos, ypos)) throw ImageBoundsError(xpos, ypos, this->_bounds);
    return this->_data[this->addressPixel(xpos, ypos)];
}

template <typename T>
ConstImageView<T> BaseImage<T>::subImage(const Bounds<int>& bounds) const 
{
    if (!this->_bounds.includes(bounds)) {
        std::ostringstream os;
        os << "Subimage bounds (" << bounds << ") are outside original image bounds (" 
           << this->_bounds << ")";
        throw ImageError(os.str());
    }
    T* newdata = _data
        + (bounds.getYMin() - this->_bounds.getYMin()) * _stride
        + (bounds.getXMin() - this->_bounds.getXMin());
    return ConstImageView<T>(newdata,_owner,_stride,bounds);
}

template <typename T>
ImageView<T> ImageView<T>::subImage(const Bounds<int>& bounds) const 
{
    if (!this->_bounds.includes(bounds)) {
        std::ostringstream os;
        os << "Subimage bounds (" << bounds << ") are outside original image bounds (" 
           << this->_bounds << ")";
        throw ImageError(os.str());
    }
    T* newdata = this->_data
        + (bounds.getYMin() - this->_bounds.getYMin()) * this->_stride
        + (bounds.getXMin() - this->_bounds.getXMin());
    return ImageView<T>(newdata,this->_owner,this->_stride,bounds);
}

/////////////////////////////////////////////////////////////////////
//// Arithmetic functions
///////////////////////////////////////////////////////////////////////

// TODO: These are currently inefficient.
// Need to change them to return composite objects that can then assignTo the 
// final image location.  i.e.
// im1 + im2 return SumOfImages(im1,im2) that doesn't do anything yet.
// Then im3 = im1 + im2 turns into SumOfImages(im1,im2).assignTo(im3)
// where the actual calculation is done once it knows where to write to.
template <typename T>
Image<T> BaseImage<T>::operator+(const BaseImage<T>& rhs) const 
{
    Image<T> result = *this;
    result += rhs;
    return result;
}

template <typename T>
Image<T> BaseImage<T>::operator-(const BaseImage<T>& rhs) const 
{
    Image<T> result = *this;
    result -= rhs;
    return result;
}

template <typename T>
Image<T> BaseImage<T>::operator*(const BaseImage<T>& rhs) const 
{
    Image<T> result = *this;
    result *= rhs;
    return result;
}

template <typename T>
Image<T> BaseImage<T>::operator/(const BaseImage<T>& rhs) const 
{
    Image<T> result = *this;
    result /= rhs;
    return result;
}

namespace {

template <typename T>
class ConstReturn 
{
public: 
    ConstReturn(const T v): val(v) {}
    T operator()(const T dummy) const { return val; }
private:
    T val;
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
const ImageView<T>& ImageView<T>::operator+=(T x) const 
{
    transform_pixel(*this, bind2nd(std::plus<T>(),x));
    return *this;
}

template <typename T>
const ImageView<T>& ImageView<T>::operator-=(T x) const 
{
    transform_pixel(*this, bind2nd(std::minus<T>(),x));
    return *this;
}

template <typename T>
const ImageView<T>& ImageView<T>::operator*=(T x) const 
{
    transform_pixel(*this, bind2nd(std::multiplies<T>(),x));
    return *this;
}

template <typename T>
const ImageView<T>& ImageView<T>::operator/=(T x) const 
{
    transform_pixel(*this, bind2nd(std::divides<T>(),x));
    return *this;
}

template <typename T>
void ImageView<T>::copyFrom(const BaseImage<T>& rhs) const
{
    if (!this->_bounds.isSameShapeAs(rhs.getBounds()))
        throw ImageError("Attempt im1 = im2, but bounds not the same shape");
    transform_pixel(*this, rhs, ReturnSecond<T>());
}

template <typename T>
const ImageView<T>& ImageView<T>::operator+=(const BaseImage<T>& rhs) const 
{
    if (!this->_bounds.isSameShapeAs(rhs.getBounds()))
        throw ImageError("Attempt im1 += im2, but bounds not the same shape");
    transform_pixel(*this, rhs, std::plus<T>());
    return *this;
}

template <typename T>
const ImageView<T>& ImageView<T>::operator-=(const BaseImage<T>& rhs) const 
{
    if (!this->_bounds.isSameShapeAs(rhs.getBounds()))
        throw ImageError("Attempt im1 -= im2, but bounds not the same shape");
    transform_pixel(*this, rhs, std::minus<T>());
    return *this;    
}

template <typename T>
const ImageView<T>& ImageView<T>::operator*=(const BaseImage<T>& rhs) const 
{
    if (!this->_bounds.isSameShapeAs(rhs.getBounds()))
        throw ImageError("Attempt im1 *= im2, but bounds not the same shape");
    transform_pixel(*this, rhs, std::multiplies<T>());
    return *this;    
}

template <typename T>
const ImageView<T>& ImageView<T>::operator/=(const BaseImage<T>& rhs) const 
{
    if (!this->_bounds.isSameShapeAs(rhs.getBounds()))
        throw ImageError("Attempt im1 /= im2, but bounds not the same shape");
    transform_pixel(*this, rhs, std::divides<T>());
    return *this;    
}

// instantiate for expected types

template class BaseImage<double>;
template class BaseImage<float>;
template class BaseImage<int>;
template class BaseImage<short>;
template class Image<double>;
template class Image<float>;
template class Image<int>;
template class Image<short>;
template class ImageView<double>;
template class ImageView<float>;
template class ImageView<int>;
template class ImageView<short>;
template class ConstImageView<short>;
template class ConstImageView<double>;
template class ConstImageView<float>;
template class ConstImageView<int>;

} // namespace galsim

