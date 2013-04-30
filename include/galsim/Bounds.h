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
/// @file Bounds.h @brief Classes defining 2d positions and rectangles.

#ifndef Bounds_H
#define Bounds_H

#include <vector>
#include <iostream>
#include <limits>

#include "Std.h"

namespace galsim {

    template <class T>
    /// @brief Class for storing 2d position vectors in an (x, y) format.
    class Position 
    {
    public:
        /// @brief Publicly visible x & y attributes of the position.
        T x,y;

        ///@brief Constructor.
        Position(const T xin=0, const T yin=0) : x(xin), y(yin) {}

        ///@brief Assignment.
        Position& operator=(const Position<T>& rhs) 
        {
            if (&rhs == this) return *this;
            else { x=rhs.x; y=rhs.y; return *this; }
        }

        /// @brief Overloaded += operator, following standard vector algebra rules.
        Position<T>& operator+=(const Position<T>& rhs) { x+=rhs.x; y+=rhs.y; return *this; }

        /// @brief Overloaded -= operator, following standard vector algebra rules.
        Position<T>& operator-=(const Position<T>& rhs) { x-=rhs.x; y-=rhs.y; return *this; }

        /// @brief Overloaded *= operator for scalar multiplication.
        Position<T>& operator*=(const T rhs) { x*=rhs; y*=rhs; return *this; }

        /// @brief Overloaded /= operator for scalar division.
        Position<T>& operator/=(const T rhs) { x/=rhs; y/=rhs; return *this; }

        /// @brief Overloaded * operator for scalar on rhs.
        Position<T> operator*(const T rhs) const { return Position<T>(x*rhs, y*rhs); }

        /// @brief Allow T * Position as well.
        friend Position<T> operator*(const T lhs, const Position<T>& rhs) { return rhs*lhs; }

        /// @brief Overloaded / operator for scalar on rhs.
        Position<T> operator/(const T rhs) const { return Position<T>(x/rhs, y/rhs); }

        /// @brief Unary negation (x, y) -> (-x, -y).
        Position<T> operator-() const { return Position<T>(-x,-y); }

        /// @brief Overloaded vector + addition operator with a Position on the rhs.
        Position<T> operator+(const Position<T>& rhs) const 
        { return Position<T>(x+rhs.x,y+rhs.y); }

        /// @brief Overloaded vector - subtraction operator with a Position on the rhs.
        Position<T> operator-(const Position<T>& rhs) const 
        { return Position<T>(x-rhs.x,y-rhs.y); }

        /// @brief Overloaded == relational equality operator.
        bool operator==(const Position<T>& rhs) const { return (x==rhs.x && y==rhs.y); }
        
        /// @brief Overloaded != relational non-equality operator.
        bool operator!=(const Position<T>& rhs) const { return (x!=rhs.x || y!=rhs.y); }

        /// @brief Write (x, y) position to output stream.
        void write(std::ostream& fout) const { fout << "(" << x << "," << y << ")"; }

        /// @brief Read (x, y) position from input istream.
        void read(std::istream& fin) { char ch; fin >> ch >> x >> ch >> y >> ch; }

    }; // Position

    /// @brief Overloaded << operator which uses write() method of Position class.
    template <class T>
    inline std::ostream& operator<<(std::ostream& os, const Position<T> p) 
    { p.write(os); return os; }

    /// @brief Overloaded >> operator which uses read() method of Position class.
    template <class T>
    inline std::istream& operator>>(std::istream& is, Position<T>& p) 
    { p.read(is); return is; }

    template <class T>
    /** 
     * @brief Class for storing image bounds, essentially the vertices of a rectangle.  
     *
     * This is used to keep track of the bounds of catalogs and fields.  You can set values, 
     * but generally you just keep including positions of each galaxy or the bounds of each 
     * catalog respectively using the += operators.
     *
     * The bounds are stored as four numbers in each instance, (xmin, ymin, xmax, ymax), with an
     * additional boolean switch to say whether or not the Bounds rectangle has been defined.
     *
     * Rectangle is undefined if min>max in either direction.
     */
    class Bounds 
    {
    public:
        /// @brief Constructor using four scalar positions (xmin, xmax, ymin, ymax).
        Bounds(const T x1, const T x2, const T y1, const T y2) :
            defined(x1<=x2 && y1<=y2),xmin(x1),xmax(x2),ymin(y1),ymax(y2) {}

        /// @brief Constructor using a single Position vector x/ymin = x/ymax.
        Bounds(const Position<T>& pos) :
            defined(1),xmin(pos.x),xmax(pos.x),ymin(pos.y),ymax(pos.y) {}

        /// @brief Constructor using two Positions, first for x/ymin, second for x/ymax.
        Bounds(const Position<T>& pos1, const Position<T>& pos2) :
            defined(1),xmin(std::min(pos1.x,pos2.x)),xmax(std::max(pos1.x,pos2.x)),
            ymin(std::min(pos1.y,pos2.y)),ymax(std::max(pos1.y,pos2.y)) {}

        /// @brief Constructor for empty Bounds, .isDefined() method will return false.
        Bounds() : defined(0),xmin(0),xmax(0),ymin(0),ymax(0) {}

        /// @brief Destructor.
        ~Bounds() {}

        /// @brief Set the xmin of the Bounds rectangle.
        void setXMin(const T x) { xmin = x; defined= xmin<=xmax && ymin<=ymax; }

        /// @brief Set the xmax of the Bounds rectangle. 
        void setXMax(const T x) { xmax = x; defined= xmin<=xmax && ymin<=ymax; }

        /// @brief Set the ymin of the Bounds rectangle.
        void setYMin(const T y) { ymin = y; defined= xmin<=xmax && ymin<=ymax; }

        /// @brief Set the ymax of the Bounds rectangle.
        void setYMax(const T y) { ymax = y; defined= xmin<=xmax && ymin<=ymax; }

        /// @brief Get the xmin of the Bounds rectangle.
        T getXMin() const { return xmin; }

        /// @brief Get the xmax of the Bounds rectangle.
        T getXMax() const { return xmax; }

        /// @brief Get the ymin of the Bounds rectangle.
        T getYMin() const { return ymin; }

        /// @brief Get the ymax of the Bounds rectangle.
        T getYMax() const { return ymax; }

        /// @brief Query whether the Bounds rectangle is defined.
        bool isDefined() const { return defined; }

        /// @brief Return the nominal center of the image. 
        ///
        /// This is the position of the pixel that is considered to be (0,0)
        Position<T> center() const;

        /// @brief Return the true center of the image.
        ///
        /// For even-sized, integer bounds, this will not be an integer, since the center in 
        /// that case falls between two pixels.
        Position<double> trueCenter() const;

        /// @brief expand bounds to include this point
        void operator+=(const Position<T>& pos);

        /// @brief expand bounds to include these bounds
        void operator+=(const Bounds<T>& rec);

        //@{
        /// @brief add a border of size d around existing bounds
        void addBorder(const T d);
        void operator+=(const T d) { addBorder(d); }
        //@}

        /// @brief expand bounds by a factor m around the current center.
        void expand(const double m); 

        /// @brief find the intersection of two bounds
        const Bounds<T> operator&(const Bounds<T>& rhs) const; 

        //@{
        /// @brief shift the bounding box by some amount.
        void shift(const T dx, const T dy) { xmin+=dx; xmax+=dx; ymin+=dy; ymax+=dy; }
        void shift(Position<T> dx) { shift(dx.x,dx.y); }
        //@}

        //@{
        /// @brief return whether the bounded region includes a given point
        bool includes(const Position<T>& pos) const
        { return (defined && pos.x<=xmax && pos.x>=xmin && pos.y<=ymax && pos.y>=ymin); }
        bool includes(const T x, const T y) const
        { return (defined && x<=xmax && x>=xmin && y<=ymax && y>=ymin); }
        //@}

        /// @brief return whether the bounded region includes all of the given bounds
        bool includes(const Bounds<T>& rhs) const
        { 
            return (defined && rhs.defined && 
                    rhs.xmin>=xmin && rhs.xmax<=xmax &&
                    rhs.ymin>=ymin && rhs.ymax<=ymax);
        }

        /// @brief check equality of two bounds
        bool operator==(const Bounds<T>& rhs) const 
        {
            return defined && rhs.defined && (xmin==rhs.xmin) &&
                (ymin==rhs.ymin) && (xmax==rhs.xmax) && (ymax==rhs.ymax);
        }
        bool operator!=(const Bounds<T>& rhs) const 
        {
            return !defined || !rhs.defined || (xmin!=rhs.xmin) ||
                (ymin!=rhs.ymin) || (xmax!=rhs.xmax) || (ymax!=rhs.ymax);
        }

        ///  @brief Check if two bounds have same shape, but maybe different origin.
        bool isSameShapeAs(const Bounds<T>& rhs) const
        {
            return defined && rhs.defined && 
                (xmax - xmin == rhs.xmax - rhs.xmin) &&
                (ymax - ymin == rhs.ymax - rhs.ymin);
        }

        /**
         *  @brief Return the area of the enclosed region
         *
         *  The area is a bit different for integer-type Bounds and float-type bounds.
         *  For floating point types, it is simply (xmax-xmin)*(ymax-ymin).
         *  However, for integer type, we need to add 1 to each size to correctly count the
         *  number of pixels being described by the bounding box.
         */
        T area() const 
        {
            return defined ?
                ( std::numeric_limits<T>::is_integer ?
                  (xmax-xmin+1)*(ymax-ymin+1) :
                  (xmax-xmin)*(ymax-ymin) ) :
                T(0); 
        }

        /// @brief divide the current bounds into (nx x ny) sub-regions
        typename std::vector<Bounds<T> > divide(int nx, int ny) const;

        /// @brief write out to a file
        void write(std::ostream& fout) const
        { 
            if (defined) fout << xmin << ' ' << xmax << ' ' << ymin << ' ' << ymax << ' ';
            else fout << "Undefined ";
        }

        /// @brief read in from a file
        void read(std::istream& fin)
        { fin >> xmin >> xmax >> ymin >> ymax; defined = xmin<=xmax && ymin<=ymax; }

    private:
        bool defined;
        T xmin,xmax,ymin,ymax;

    };

    template <class T>
    inline std::ostream& operator<<(std::ostream& fout, const Bounds<T>& b)
    { b.write(fout); return fout; }

    template <class T>
    inline std::istream& operator>>(std::istream& fin, Bounds<T>& b)
    { b.read(fin); return fin; }

    ///////////////////////////////////////////////////////////////////////
    //  Following are the implementations:
    ///////////////////////////////////////////////////////////////////////

    template <class T>
    void Bounds<T>::operator+=(const Position<T>& pos)
    {
        if (defined) {
            if(pos.x < xmin) xmin = pos.x;
            else if (pos.x > xmax) xmax = pos.x;
            if(pos.y < ymin) ymin = pos.y;
            else if (pos.y > ymax) ymax = pos.y;
        } else {
            xmin = xmax = pos.x;
            ymin = ymax = pos.y;
            defined = 1;
        }
    }

    template <class T>
    void Bounds<T>::operator+=(const Bounds<T>& rec)
    {
        if (!rec.isDefined()) return;
        if (defined) {
            if(rec.getXMin() < xmin) xmin = rec.getXMin();
            if(rec.getXMax() > xmax) xmax = rec.getXMax();
            if(rec.getYMin() < ymin) ymin = rec.getYMin();
            if(rec.getYMax() > ymax) ymax = rec.getYMax();
        } else {
            *this = rec;
            defined = 1;
        }
    }

    // First the generic version:
    template <class T, class U, bool is_int>
    struct CalculateCenter 
    {
        static Position<U> call(const Bounds<T>& b)
        { return Position<U>((b.getXMin() + b.getXMax())/U(2),(b.getYMin() + b.getYMax())/U(2)); }
    };
    // Slightly different for integer types:
    template <class T, class U>
    struct CalculateCenter<T, U, true>
    {
        static Position<U> call(const Bounds<T>& b)
        { return Position<U>((b.getXMin()+b.getXMax()+1)/U(2),(b.getYMin()+b.getYMax()+1)/U(2)); }
    };

    template <class T>
    Position<T> Bounds<T>::center() const
    { return CalculateCenter<T,T,std::numeric_limits<T>::is_integer>::call(*this); }

    template <class T>
    Position<double> Bounds<T>::trueCenter() const
    { return CalculateCenter<T,double,false>::call(*this); }

    // & operator finds intersection, if any
    template <class T>
    const Bounds<T> Bounds<T>::operator&(const Bounds<T>& rhs) const
    {
        if (!defined || !rhs.defined) return Bounds<T>();
        Bounds<T> temp(
            xmin<rhs.xmin ? rhs.xmin : xmin,
            xmax>rhs.xmax ? rhs.xmax : xmax,
            ymin<rhs.ymin ? rhs.ymin : ymin,
            ymax>rhs.ymax ? rhs.ymax : ymax);
        if (temp.xmin>temp.xmax || temp.ymin>temp.ymax) return Bounds<T>();
        else return temp;
    }

    template <class T>
    void Bounds<T>::addBorder(T d)
    { if(defined) { xmax += d; xmin -= d; ymax += d; ymin -= d; } }

    template <class T>
    std::vector< Bounds<T> > Bounds<T>::divide(int nx, int ny) const
    {
        if (!defined) return std::vector< Bounds<T> >(nx,ny);
        typename std::vector< Bounds<T> > temp(nx*ny);
        std::vector<double> x(nx+1);
        std::vector<double> y(ny+1);
        x[0] = xmin;  x[nx] = xmax;
        y[0] = ymin;  y[ny] = ymax;
        double xstep = (xmax-xmin)/nx;
        double ystep = (ymax-ymin)/ny;
        for(int i=1;i<nx;i++) x[i] = x[0]+i*xstep;
        for(int j=1;j<ny;j++) y[j] = y[0]+j*ystep;
        typename std::vector< Bounds<T> >::iterator ii=temp.begin();
        for(int i=0;i<nx;i++) for(int j=0;j<ny;j++, ++i)
            *ii = Bounds<T>(x[i],x[i+1],y[j],y[j+1]);
        return temp;
    }

    template <class T>
    void Bounds<T>::expand(const double m) 
    {
        T dx = xmax-xmin;
        T dy = ymax-ymin;
        dx = T(dx*0.5*(m-1.)); 
        dy = T(dy*0.5*(m-1.));
        xmax += dx;  xmin -= dx;
        ymax += dy;  ymin -= dy;
    }

}

#endif
