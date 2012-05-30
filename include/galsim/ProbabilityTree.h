// -*- c++ -*-
#ifndef PROBABILITY_TREE_H
#define PROBABILITY_TREE_H

#include <list>

namespace galsim {

    /** 
     * @brief Class to build binary tree for random draw among objects with known probabilities
     *
     * The class is derived from a list of objects of any type T.
     * The class T can be anything that has a `getFlux()` call.  The absolute value of
     * the return from `getFlux()` is taken as the relative probability that should be assigned
     * to this member of the list.  The purpose of this class is to select a member of the
     * list given a uniform random number in the interval [0,1).  This is what the `find()`
     * member does.
     *
     * This works by creating a binary search tree with less-probable members being at
     * successively deeper levels of the tree.  Each member is assigned an interval of the
     * cumulative probability (a.k.a. flux) distribution, in the order of walking the tree,
     * and the `find()` method simply descends the tree looking for the interval containing
     * the specified random number.
     *
     * To use the class, just append your members to this class using the std::list 
     * methods.  Then call `buildTree()`, optionally specifying a minimum level of flux
     * for members to be retained in the tree (default is that any non-zero member is in).
     * The `find()` method will now return random draws with near-optimal speed.
     */
    template <class T>
    class ProbabilityTree: private std::list<T> {
    public:
        using std::list<T>::begin;
        using std::list<T>::end;
        using std::list<T>::push_back;
        using std::list<T>::splice;
        using std::list<T>::empty;
        using std::list<T>::clear;

        /// @brief Constructor - nothing to do.
        ProbabilityTree(): root(0) {};
        /// @brief Destructor - kill the `Element`s that have been stored away
        ~ProbabilityTree() {flushElements();}
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
        T* find(double& unitRandom) const {
            if (!root) throw std::runtime_error("ProbabilityTree::find() before buildTree();");
            unitRandom *= totalAbsoluteFlux;
            return root->find(unitRandom);
        }
        /** 
         * @brief Construct the tree from current list elements.
         * @param[in] threshold that have flux <= this value are not included in the tree.
         */
        void buildTree(double threshold=0.) {
            if (std::list<T>::empty()) return;
            flushElements();

            // Sort the list by absolute value of flux;
            this->sort(compare);

            // Construct tree by adding members to levels in flux order.
            // nextChild points to next list member to be added to tree at leaf level.
            typename std::list<T>::iterator nextChild = std::list<T>::begin();
            root = new Element(*nextChild);
            elementStorage.push_back(root);
            ++nextChild;
            // nextParent points to the first Element that has room for another child.
            typename std::list<Element*>::iterator nextParent = elementStorage.begin();
            while (nextChild != std::list<T>::end()) {
                if ( std::abs(nextChild->getFlux()) <= threshold) {
                    // Skip this member if it does not have enough flux
                    ++nextChild;
                    continue;
                }
                Element* child = new Element(*nextChild);
                ++nextChild;
                elementStorage.push_back(child);
                if ((*nextParent)->left) {
                    // Have a left child already, give the parent a right child and
                    // advance parent pointer to next object.
                    (*nextParent)->right = child;
                    ++nextParent;
                } else {
                    (*nextParent)->left = child;
                }
            }

            // Walk tree to build cumulative flux distribution.
            totalAbsoluteFlux = 0.;
            accumulateFlux(root);
        }
    private:
        /// @brief A private class that wraps the members in their tree information
        class Element {
        public:
            Element(T& data): dataPtr(&data), left(0), right(0) {}
            T* dataPtr; ///< Pointer to the member for this element
            Element* left; ///< Pointer to left child member
            Element* right; ///< Pointer to right child member
            double differentialFlux; ///< The unnormalized probability in this element
            /// Total unnorm. probability of all elements before this one in tree
            double leftCumulativeFlux;
            /// Total unnorm. probability of all elements before & including this one
            double rightCumulativeFlux;
            /**
             * Recursive routine to find Element that contains a given value
             * in the cumulative flux (unnormalized probability) distribution.
             * @param[in,out] cumulativeFlux On input, randomly chosen point on cumulative
             *  flux distribution.  On output: fraction of chosen member's flux that is
             *  below the input value on cumulative flux distribution.
             * @returns pointer to member that contains input cumulative flux point.
             */
            T* find(double& cumulativeFlux) const {
                if (left && (cumulativeFlux < leftCumulativeFlux)) 
                    return left->find(cumulativeFlux);
                else if (right && (cumulativeFlux > rightCumulativeFlux))
                    return right->find(cumulativeFlux);
                // Answer is this element.
                cumulativeFlux = (cumulativeFlux - leftCumulativeFlux) / differentialFlux;
                return dataPtr;
            }
        };
        std::list<Element*> elementStorage; ///< Container for all Elements created.
        Element* root;  ///< root of the tree;
        double totalAbsoluteFlux; ///< Stored total unnormalized probability

        /// @brief Cleanup created objects
        void flushElements() {
            for (typename std::list<Element*>::iterator i=elementStorage.begin();
                 i != elementStorage.end();
                 ++i)
                delete *i;
            elementStorage.clear();
            root = 0;
        }

        /// @brief Recursive routine to walk the tree and assign cumulative fluxes to Elements
        void accumulateFlux(Element* el) {
            // Accumulate any flux in objects at/below left node
            if (el->left) accumulateFlux(el->left);
            // Note cumulative flux before/after this node, increment totalAbsoluteFlux
            el->leftCumulativeFlux = totalAbsoluteFlux;
            el->differentialFlux = std::abs(el->dataPtr->getFlux());
            totalAbsoluteFlux += el->differentialFlux;
            el->rightCumulativeFlux = totalAbsoluteFlux;
            // Accumulate any flux in objects at/below right node.
            if (el->right) accumulateFlux(el->right);
        }

        /// @brief Comparison class to sort inputs in *descending* flux order.
        class FluxCompare {
        public:
            bool operator()(const T& lhs, const T& rhs) const {
                return std::abs(lhs.getFlux()) > std::abs(rhs.getFlux());
            }
        } compare;
    };
} // end namespace galsim

#endif
