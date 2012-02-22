// $Id: Image2.h,v 1.6 2009/11/02 22:48:53 garyb Exp $
// Further pieces of the Image.h header.  Here are classes that
// are going to be used directly by the programmer.
#ifndef IMAGE2_H
#define IMAGE2_H

namespace img {
  using namespace std;

  class ImageError;
  class ImageBounds;

  ////////////////////////////////////////////////////////////////
  // The pixel data structure - never used by outside programs
  ////////////////////////////////////////////////////////////////
  template <class T=float>
  class ImageData {
    //    template <class U>
    //friend class FITSImage;
  public:
    // Create:
    // image with unspecified data values:
    ImageData(const Bounds<int> inBounds) ;
    // image filled with a scalar:
    ImageData(const Bounds<int> inBounds, const T initValue) ;
    // image for which the data array has been set up by someone else:
    ImageData(const Bounds<int> inBounds, 
	      T** rptrs,
	      bool _contig=false);
  
    ~ImageData();

    // This routine used by some other object rearranging the storage
    // array.  If ownRowPointers was set, then this routine deletes the
    // old array and assumes responsibility for deleting the new one.
    void replaceRowPointers(T **newRptrs) const {
      if (ownRowPointers) delete[] (rowPointers + bounds.getYMin()); 
      rowPointers=newRptrs;
    }
    // Make a new ImageData that is subimage of this one.  Data will be
    // shared in memory, just pixel bounds are different.  subimages
    // a.k.a. children must be destroyed before the parent.
    ImageData* subimage(const Bounds<int> bsub);
    const ImageData* const_subimage(const Bounds<int> bsub) const;  //return const
    const ImageData* subimage(const Bounds<int> bsub) const;  //if this const
    void deleteSubimages() const; // delete all living subimages

    // Create a new subimage that is duplicate of this ones data
    ImageData* duplicate() const;
  
    // Resize the image - this will throw exception if data array
    // is not owned or if there are subimages in use.
    // Data are undefined afterwards
    void resize(const Bounds<int> newBounds);
    // Copy in the data (and size) from another image.
    void copyFrom(const ImageData<T>& rhs);

    // move origin of coordinates, preserving data.  Throws exception
    // if data are not owned or if there are subimages in use.
    void shift(int x0, int y0);

    // Element access - unchecked
    const T& operator()(const int xpos, const int ypos) const {
      return *location(xpos,ypos);
    }
    T& operator()(const int xpos, const int ypos) {
      return *location(xpos,ypos);
    }

    // Element access - checked
    const T& at(const int xpos, const int ypos) const {
      if (!bounds.includes(xpos,ypos))  
      	ImageBounds::FormatAndThrow(xpos,ypos,bounds);
      return *location(xpos,ypos);
    }
    T& at(const int xpos, const int ypos) {
      if (!bounds.includes(xpos,ypos))  
	ImageBounds::FormatAndThrow(xpos,ypos,bounds);
      return *location(xpos,ypos);
    }

    // give pointer to a pixel in the storage array, 
    // for use by routines that buffer image data for us.
    T*   location(const int xpos, const int ypos) const {
      return *(rowPointers+ypos)+xpos;}

    // Access functions
    Bounds<int> getBounds() const {return bounds;}
    bool contiguousData() const {return isContiguous;}

  private:
    // image which will be a subimage of a parent:
    ImageData(const Bounds<int> inBounds, 
	      const ImageData<T>* _parent);
    ImageData(const ImageData &) {
      throw ImageError("Attempt to use ImageData copy constructor");
    }	//No inadvertent copying allowed! Use copyFrom() to be explicit.
    ImageData& operator=(const ImageData&) {
      throw ImageError("Attempt to use ImageData operator=");
    }

    Bounds<int>	bounds;
    mutable T	**rowPointers;	// Pointers to start of the data rows
    const ImageData<T>* parent; // If this is a subimage, what's parent?
    //list of subimages of this (sub)image:
    mutable list<const ImageData<T>*> children;	

    // Does this object own (i.e. have responsibility for destroying):
    bool  ownDataArray;	// the actual data array
    bool  ownRowPointers;	// the rowpointer array
    mutable bool  isContiguous;	// Set if entire image is contiguous in memory

    // class utility functions:
    void acquireArrays(Bounds<int> inBounds);
    void discardArrays();
    void unlinkChild(const ImageData<T>* child) const;
    void linkChild(const ImageData<T>* child) const;
  };

  //////////////////////////////////////////////////////////////////////////
  // Checked iterator for images
  //////////////////////////////////////////////////////////////////////////
  // ??? need conversion from iter to const iter??
  template <class T=float>
  class ImageChk_iter {
  private:
    ImageData<T> *I;
    T* ptr;
    int col;	//keep track of column number
  public:
    ImageChk_iter(ImageData<T>* ii, const int x, const int y): I(ii), col(x) {
      if (y<I->getBounds().getYMin() || y>I->getBounds().getYMax())
	ImageBounds::FormatAndThrow("row",
				    I->getBounds().getYMin(), I->getBounds().getYMax(),
				    y);
      ptr = &( (*I)(x,y));
    }
    T& operator*() const {
      if (col<I->getBounds().getXMin() || col>I->getBounds().getXMax())
	ImageBounds::FormatAndThrow("column",
				    I->getBounds().getXMin(), I->getBounds().getXMax(),
				    col);
      return *ptr;
    }
    ImageChk_iter operator++() {++ptr; ++col; return *this;}
    ImageChk_iter operator--() {++ptr; ++col; return *this;}
    ImageChk_iter operator+=(int i) {ptr+=i; col+=i; return *this;}
    ImageChk_iter operator-=(int i) {ptr-=i; col-=i; return *this;}
    bool operator<(const ImageChk_iter rhs) const {return ptr<rhs.ptr;}
    bool operator<=(const ImageChk_iter rhs) const {return ptr<=rhs.ptr;}
    bool operator>(const ImageChk_iter rhs) const {return ptr>rhs.ptr;}
    bool operator>=(const ImageChk_iter rhs) const {return ptr>=rhs.ptr;}
    bool operator==(const ImageChk_iter rhs) const {return ptr==rhs.ptr;}
    bool operator!=(const ImageChk_iter rhs) const {return ptr!=rhs.ptr;}
  };

  template <class T=float>
  class ImageChk_citer {
  private:
    const ImageData<T> *I;
    const T* ptr;
    int col;	//keep track of column number
  public:
    ImageChk_citer(const ImageData<T>* ii, 
		   const int x, const int y): I(ii), col(x) {
      if (y<I->getBounds().getYMin() || y>I->getBounds().getYMax())
	ImageBounds::FormatAndThrow("row",
				    I->getBounds().getYMin(), I->getBounds().getYMax(),
				    y);
      ptr = &( (*I)(x,y));
    }
    const T& operator*() const {
      if (col<I->getBounds().getXMin() || col>I->getBounds().getXMax())
	ImageBounds::FormatAndThrow("column",
				    I->getBounds().getXMin(), I->getBounds().getXMax(),
				    col);
      return *ptr;
    }
    ImageChk_citer operator++() {++ptr; ++col; return *this;}
    ImageChk_citer operator--() {++ptr; ++col; return *this;}
    ImageChk_citer operator+=(int i) {ptr+=i; col+=i; return *this;}
    ImageChk_citer operator-=(int i) {ptr-=i; col-=i; return *this;}
    bool operator<(const ImageChk_citer rhs) const {return ptr<rhs.ptr;}
    bool operator<=(const ImageChk_citer rhs) const {return ptr<=rhs.ptr;}
    bool operator>(const ImageChk_citer rhs) const {return ptr>rhs.ptr;}
    bool operator>=(const ImageChk_citer rhs) const {return ptr>=rhs.ptr;}
    bool operator==(const ImageChk_citer rhs) const {return ptr==rhs.ptr;}
    bool operator!=(const ImageChk_citer rhs) const {return ptr!=rhs.ptr;}
  };

} //namespace img
#endif //IMAGE2_H
