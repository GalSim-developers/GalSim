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

#ifndef PROBABILITY_TREE_H
#define PROBABILITY_TREE_H

#include <vector>

namespace galsim {

    /** 
     * @brief Class to build binary tree for random draw among objects with known probabilities
     *
     * The class is derived from a vector of objects of any type FluxData.
     * The class FluxData can be anything that has a `getFlux()` call.  The absolute value of
     * the return from `getFlux()` is taken as the relative probability that should be assigned
     * to this member of the vector.  The purpose of this class is to select a member of the
     * vector given a uniform random number in the interval [0,1).  This is what the `find()`
     * member does.
     *
     * This works by creating a binary search tree with less-probable members being at
     * successively deeper levels of the tree.  Each member is assigned an interval of the
     * cumulative probability (a.k.a. flux) distribution, in the order of walking the tree,
     * and the `find()` method simply descends the tree looking for the interval containing
     * the specified random number.
     *
     * To use the class, just append your members to this class using the std::vector 
     * methods.  Then call `buildTree()`, optionally specifying a minimum level of flux
     * for members to be retained in the tree (default is that any non-zero member is in).
     * The `find()` method will now return random draws with near-optimal speed.
     */
    template <class FluxData>
    class ProbabilityTree :
        //! @cond  This keeps doxygen from adding vector to our list of classes.
        private std::vector<FluxData> 
        //! @endcond
    {
        typedef typename std::vector<FluxData>::iterator VecIter;
        class FluxCompare;
    public:
        using std::vector<FluxData>::size;
        using std::vector<FluxData>::begin;
        using std::vector<FluxData>::end;
        using std::vector<FluxData>::push_back;
        using std::vector<FluxData>::insert;
        using std::vector<FluxData>::empty;
        using std::vector<FluxData>::clear;

        /// @brief Constructor - nothing to do.
        ProbabilityTree() : _root(0) {}

        /// @brief Destructor - kill the `Element`s that have been stored away
        ~ProbabilityTree() { if (_root) delete _root; }

        /**
         * @brief Choose a member of the tree based on a uniform deviate
         *
         * The parameter unitRandom must be a uniform deviate in [0,1) interval.
         * On output this parameter is replaced by another random value that is
         * nominally in [0,1) interval (may be outside due to rounding errors),
         * which is actually the fraction of the winning member's flux that
         * was "inside" the point where the cumulative flux was equal to input
         * unitRandom.
         *
         * @param[in,out] unitRandom On input, a random number between 0 and 1.  On output,
         *               holds a new uniform deviate.
         * @returns Pointer to the selected tree member.
         */
        const FluxData* find(double& unitRandom) const 
        {
            // Note: Don't need floor here, since rhs is positive, so floor is superfluous.
            int i = int(unitRandom * _shortcut.size());
            assert(i < int(_shortcut.size()));
            assert(_shortcut[i]);
            unitRandom *= _totalAbsFlux;
            return _shortcut[i]->find(unitRandom);
        }

        /** 
         * @brief Construct the tree from current vector elements.
         * @param[in] threshold that have flux <= this value are not included in the tree.
         */
        void buildTree(double threshold=0.)
        {
            dbg<<"buildTree\n";
            assert(!empty());
            assert(!_root);
            // Sort the list so the largest flux regions are first.
            std::sort(begin(), end(), FluxCompare());
            VecIter last = 
                threshold == 0. ? end() :
                std::upper_bound(begin(), end(), threshold, FluxCompare());
            const int nelem = last-begin();
            dbg<<"N elements to build tree with = "<<nelem<<std::endl;
            // Figure out what the total absolute flux is
            _totalAbsFlux = 0.;
            for (VecIter it=begin(); it!=last; ++it) 
                _totalAbsFlux += std::abs(it->getFlux());
            dbg<<"totalAbsFlux = "<<_totalAbsFlux<<std::endl;
            // leftAbsFlux will be updated for each element to be the total flux up the that one.
            double leftAbsFlux = 0.;
            _root = new Element(begin(), last, leftAbsFlux, _totalAbsFlux);
            xdbg<<"Finished making root.\n";
            xdbg<<"leftAbs = "<<leftAbsFlux<<", tot = "<<_totalAbsFlux<<std::endl;
            xdbg<<"abs(diff) = "<<std::abs(leftAbsFlux - _totalAbsFlux)<<std::endl;
            xdbg<<"cf. "<<1.e-8 * _totalAbsFlux<<std::endl;
            xassert(std::abs(leftAbsFlux - _totalAbsFlux) < 1.e-8 * _totalAbsFlux);
            dbg<<"Done buildTree\n";

            // shortcut is a quick way to get to the right Element, or at least a better
            // starting point, rather than always starting at root.
            // We build this as we build the tree in the Element constructors.
            _shortcut.resize(nelem,0);
            buildShortcut(_root, 0, nelem);
            // Make sure all the shortcut entries were set.
            for(int i=0;i<nelem;++i) xassert(_shortcut[i]);
        }

    private:

        /// @brief A private class that wraps the members in their tree information
        class Element 
        {
        public:
            Element(VecIter start, VecIter end, double& leftAbsFlux, double absFlux) :
                _dataPtr(0), _left(0), _right(0),
                _leftAbsFlux(leftAbsFlux), _absFlux(absFlux), _invAbsFlux(1./absFlux)
            {
                xassert(start != end);
                if (start + 1 == end) {
                    // Only one element.
                    _dataPtr = &(*start);
                    // absFlux on input should equal the absolute flux in this dataPtr.
                    xassert(std::abs(std::abs(_dataPtr->getFlux()) - absFlux) < 
                            1.e-8 * (leftAbsFlux+absFlux));
                    // Update the running total of leftAbsFlux.
                    leftAbsFlux += _absFlux;
                } else {
                    xassert(end >= start+2);
                    VecIter mid = start;
                    // Divide the range by probability, not by number.
                    // The tree is intentionally unbalanced, so most of the time, the search
                    // stops quickly with the large flux Elements on the left.
                    double half_tot = absFlux/2.;
                    double leftSum=0.;
                    for (; leftSum <= half_tot; ++mid) leftSum += std::abs(mid->getFlux());
                    if (mid == end) {
                        // Shouldn't happen in exact arithmetic, but just in case...
                        --mid;
                        leftSum -= std::abs(mid->getFlux());
                    }
                    xassert(mid != start);
                    xassert(mid != end);
                    _left = new Element(start, mid, leftAbsFlux, leftSum);
                    _right = new Element(mid, end, leftAbsFlux, absFlux - leftSum);
                    // absFlux on input should equal the sum of the two children's fluxes.
                    xassert(std::abs((_left->_absFlux + _right->_absFlux) - absFlux) < 
                            1.e-8 * (leftAbsFlux+absFlux));
                }
            }

            ~Element() 
            {
                if (_left) {
                    assert(_right);
                    delete _left;
                    delete _right; 
                }
            }

            /**
             * Recursive routine to find Element that contains a given value
             * in the cumulative flux (unnormalized probability) distribution.
             * @param[in,out] cumulativeFlux On input, randomly chosen point on cumulative
             *  flux distribution.  On output: fraction of chosen member's flux that is
             *  below the input value on cumulative flux distribution.
             * @returns pointer to member that contains input cumulative flux point.
             */
            const FluxData* find(double& cumulativeFlux) const 
            {
                xassert(cumulativeFlux >= _leftAbsFlux);
                xassert(cumulativeFlux <= _leftAbsFlux + _absFlux);
                if (!_left) {
                    xassert(!_right);
                    // This is a leaf.  Answer is this element.
                    cumulativeFlux = (cumulativeFlux - _leftAbsFlux) * _invAbsFlux;
                    xassert(cumulativeFlux >= 0.);
                    xassert(cumulativeFlux <= 1.);
                    return _dataPtr;
                } else {
                    xassert(_right);
                    if (cumulativeFlux < _right->_leftAbsFlux)
                        return _left->find(cumulativeFlux);
                    else
                        return _right->find(cumulativeFlux);
                }
            }

            double getAbsFlux() const { return _absFlux; }
            double getLeftAbsFlux() const { return _leftAbsFlux; }
            const Element* getLeft() const { return _left; }
            const Element* getRight() const { return _right; }
            const FluxData* getData() const { return _dataPtr; }

            bool isNode() const { return bool(_left); }
            bool isLeaf() const { return !isNode(); }

        private:

            // Each Element has either a dataPtr (if it is a leaf) or left/right (if it is a node)
            const FluxData* _dataPtr; ///< Pointer to the member for this element
            Element* _left; ///< Pointer to left child member
            Element* _right; ///< Pointer to right child member

            /// Total unnorm. probability of all elements before this one in tree
            double _leftAbsFlux;

            double _absFlux; ///< The unnormalized probability in this element
            double _invAbsFlux; ///< 1./_absFlux

        };

        /// @brief Comparison class to sort inputs in *descending* flux order.
        class FluxCompare 
        {
        public:
            bool operator()(const FluxData& lhs, const FluxData& rhs) const 
            { return std::abs(lhs.getFlux()) > std::abs(rhs.getFlux()); }
            bool operator()(const FluxData& lhs, double val) const 
            { return std::abs(lhs.getFlux()) > val; }
            bool operator()(double val, const FluxData& lhs) const 
            { return val > std::abs(lhs.getFlux()); }
        };
  
        void buildShortcut(const Element* element, int i1, int i2)
        {
            // If i1 == i2, then we've already assigned everything, so stop recursing.
            if (i1 == i2) return;

            // Figure out which bins in the shortcut vector should point to this element.
            // On input, we are tasked with assigning indices i1 <= i < i2 to be either
            // this element or one of its decendents.
            xassert(i1*_totalAbsFlux/_shortcut.size() >= element->getLeftAbsFlux()-1.e-8);
            xassert(i2*_totalAbsFlux/_shortcut.size() <= 
                    element->getLeftAbsFlux()+element->getAbsFlux()+1.e-8);

            // If this is a node, then the only one we should assign is the shortcut
            // bin that include both left and right.  In other words the bin corresponding
            // to the dividing flux.
            if (element->isNode()) {
                double f = element->getRight()->getLeftAbsFlux();
                int imid = int(f * _shortcut.size() / _totalAbsFlux);
                if (imid < i1) {
                    // Then the appropriate range is all in the right subtree
                    buildShortcut(element->getRight(), i1, i2);
                } else if (imid >= i2) {
                    // Then the appropriate range is all in the left subtree
                    buildShortcut(element->getLeft(), i1, i2);
                } else {
                    // Then there is an unassigned bin that spans both children.
                    // Set it to this element.
                    assert(imid >= i1);
                    assert(imid < i2);
                    _shortcut[imid] = element;
                    // Continue on with the sub-ranges on each side.
                    buildShortcut(element->getLeft(), i1, imid);
                    buildShortcut(element->getRight(), imid+1, i2);
                }
            } else {
                // If we are at a leaf, then this leaf encompasses all the bins in the range.
                // Assigne them all to this element.
                for(int i=i1; i<i2; ++i) _shortcut[i] = element; 
            }
        }

        Element* _root;  ///< root of the tree;
        double _totalAbsFlux; ///< Stored total unnormalized probability

        /// A quicker way to get to a good starting point for find, rather than always
        /// starting with root.
        /// For a probability p, a good starting point is _shortcut[int(p*100)].
        std::vector<const Element*> _shortcut;
    };

} // end namespace galsim

#endif
