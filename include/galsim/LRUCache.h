/* -*- c++ -*-
 * Copyright (c) 2012-2017 by the GalSim developers team on GitHub
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

#ifndef GalSim_LRUCache_H
#define GalSim_LRUCache_H

#include <list>
#include <map>
#include <boost/tuple/tuple.hpp>
#include <boost/tuple/tuple_comparison.hpp>  // Need this for t1 < t2

namespace galsim {


    // Helper to build a Value from a Key
    // Normal case is that the Value take Key as a single parameter
    template <typename Value, typename Key>
    struct LRUCacheHelper
    {
        static Value* NewValue(const Key& key)
        { return new Value(key); }
    };

    // Special case of a pair, in which case the Value takes 2 parameters
    template <typename Value, typename Key1, typename Key2>
    struct LRUCacheHelper<Value,std::pair<Key1,Key2> >
    {
        static Value* NewValue(const std::pair<Key1,Key2>& key)
        { return new Value(key.first, key.second); }
    };

    // Special first few tuple cases
    template <typename Value, typename Key1>
    struct LRUCacheHelper<Value,boost::tuple<Key1> >
    {
        static Value* NewValue(const boost::tuple<Key1>& key)
        { return new Value(boost::get<0>(key)); }
    };

    template <typename Value, typename Key1, typename Key2>
    struct LRUCacheHelper<Value,boost::tuple<Key1,Key2> >
    {
        static Value* NewValue(const boost::tuple<Key1,Key2>& key)
        { return new Value(boost::get<0>(key), boost::get<1>(key)); }
    };

    template <typename Value, typename Key1, typename Key2, typename Key3>
    struct LRUCacheHelper<Value,boost::tuple<Key1,Key2,Key3> >
    {
        static Value* NewValue(const boost::tuple<Key1,Key2,Key3>& key)
        { return new Value(boost::get<0>(key), boost::get<1>(key), boost::get<2>(key)); }
    };

    template <typename Value, typename Key1, typename Key2, typename Key3, typename Key4>
    struct LRUCacheHelper<Value,boost::tuple<Key1,Key2,Key3,Key4> >
    {
        static Value* NewValue(const boost::tuple<Key1,Key2,Key3,Key4>& key)
        {
            return new Value(boost::get<0>(key), boost::get<1>(key), boost::get<2>(key),
                             boost::get<3>(key));
        }
    };

    /**
     * @brief Least Recently Used Cache
     *
     * Saves the N most recently used Values indexed by the Keys.  i.e. when it needs to remove
     * an item from the cache, it removes the _Least_ recently used item.  Whence the name.
     * c.f. http://en.wikipedia.org/wiki/Cache_algorithms#Least_Recently_Used
     *
     * The Value type should be constructible from a Key type.  e.g.
     *
     *    Key key;
     *    Value* value = new Value(key);
     *
     * Special: if Key is a std::pair<Key1, Key2>, then value takes 2 args:
     *
     *    std::pair<Key1,Key2> key = std::make_pair(key1,key2);
     *    Value* value = new Value(key1,key2);
     *
     * This structure will first look to see if we have already build such a Value given a
     * provided Key, and return it if it is in the cache.  Otherwise, it builds a new Value,
     * saves it in the cache, and returns it.
     *
     * At most nmax items will be saved in the cache.
     *
     */
    template <typename Key, typename Value>
    class LRUCache
    {
    public:
        /**
         * @brief Constructor
         *
         * @param[in] nmax  How many values to save in the cache.
         */
        LRUCache(size_t nmax) : _nmax(nmax) {}

        /**
         * @brief Destructor
         *
         * Delete all items stored in the cache.
         */
        ~LRUCache() {}

        boost::shared_ptr<Value> get(const Key& key)
        {
            assert(_entries.size() == _cache.size());
            MapIter iter = _cache.find(key);
            if (iter != _cache.end()) {
                // Item is cached.
                // Move it to the front of the list.
                if (iter != _cache.begin())
                    _entries.splice(_entries.begin(), _entries, iter->second);
                // Return the item's value
                assert(_entries.size() == _cache.size());
                return iter->second->second;
            } else {
                // Item is not cached.
                // Make a new one.
                boost::shared_ptr<Value> value(LRUCacheHelper<Value,Key>::NewValue(key));
                // Remove items from the cache as necessary.
                while (_entries.size() >= _nmax) {
                    bool erased = _cache.erase(_entries.back().first);
                    assert(erased);
                    _entries.pop_back();
                }
                // Add the new value to the front.
                _entries.push_front(Entry(key,value));
                // Also put it in the cache
                _cache[key] = _entries.begin();
                // Return the new value
                assert(_entries.size() == _cache.size());
                return value;
            }
        }

    private:

        size_t _nmax;

        typedef std::pair<Key, boost::shared_ptr<Value> > Entry;
        std::list<Entry> _entries;

        typedef typename std::list<Entry>::iterator ListIter;
        std::map<Key, ListIter> _cache;

        typedef typename std::map<Key, ListIter>::iterator MapIter;
    };

}

#endif

