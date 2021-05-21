/* -*- c++ -*-
 * Copyright (c) 2012-2021 by the GalSim developers team on GitHub
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

namespace galsim {

    // A very simple tuple class that just does what we need for the LRUCache.
    // It can hold up to a maximum of 5 parameters.
    template <typename T1, typename T2=int, typename T3=int, typename T4=int, typename T5=int>
    class Tuple
    {
    public:
        T1 first;
        T2 second;
        T3 third;
        T4 fourth;
        T5 fifth;

        Tuple(const T1& a) : first(a), second(0), third(0), fourth(0), fifth(0) {}
        Tuple(const T1& a, const T2& b) : first(a), second(b), third(0), fourth(0), fifth(0) {}
        Tuple(const T1& a, const T2& b, const T3& c) :
            first(a), second(b), third(c), fourth(0), fifth(0) {}
        Tuple(const T1& a, const T2& b, const T3& c, const T4& d) :
            first(a), second(b), third(c), fourth(d), fifth(0) {}
        Tuple(const T1& a, const T2& b, const T3& c, const T4& d, const T5& e) :
            first(a), second(b), third(c), fourth(d), fifth(e) {}

        Tuple(const Tuple& rhs) :
            first(rhs.first), second(rhs.second), third(rhs.third), fourth(rhs.fourth),
            fifth(rhs.fifth) {}

        Tuple& operator=(const Tuple& rhs)
        {
            if (&rhs != this) {
                first = rhs.first;
                second = rhs.second;
                third = rhs.third;
                fourth = rhs.fourth;
                fifth = rhs.fifth;
            }
            return *this;
        }

        bool operator<(const Tuple& rhs) const
        {
            return (
                first < rhs.first ? true :
                rhs.first < first ? false :
                second < rhs.second ? true :
                rhs.second < second ? false :
                third < rhs.third ? true :
                rhs.third < third ? false :
                fourth < rhs.fourth ? true :
                rhs.fourth < fourth ? false :
                fifth < rhs.fifth ? true :
                false);
        }
    };

    template <typename T1>
    Tuple<T1> MakeTuple(const T1& a)
    { return Tuple<T1>(a); }
    template <typename T1, typename T2>
    Tuple<T1,T2> MakeTuple(const T1& a, const T2& b)
    { return Tuple<T1,T2>(a,b); }
    template <typename T1, typename T2, typename T3>
    Tuple<T1,T2,T3> MakeTuple(const T1& a, const T2& b, const T3& c)
    { return Tuple<T1,T2,T3>(a,b,c); }
    template <typename T1, typename T2, typename T3, typename T4>
    Tuple<T1,T2,T3,T4> MakeTuple(const T1& a, const T2& b, const T3& c, const T4& d)
    { return Tuple<T1,T2,T3,T4>(a,b,c,d); }
    template <typename T1, typename T2, typename T3, typename T4, typename T5>
    Tuple<T1,T2,T3,T4,T5> MakeTuple(const T1& a, const T2& b, const T3& c, const T4& d, const T5& e)
    { return Tuple<T1,T2,T3,T4,T5>(a,b,c,d,e); }

    // Helper to build a Value from a Key
    // Normal case is that the Value take Key as a single parameter
    template <typename Value, typename Key>
    struct LRUCacheHelper
    {
        static Value* NewValue(const Key& key)
        { return new Value(key); }
    };

    // Special first few tuple cases
    template <typename Value, typename Key1>
    struct LRUCacheHelper<Value,Tuple<Key1> >
    {
        static Value* NewValue(const Tuple<Key1>& key)
        { return new Value(key.first); }
    };

    template <typename Value, typename Key1, typename Key2>
    struct LRUCacheHelper<Value,Tuple<Key1,Key2> >
    {
        static Value* NewValue(const Tuple<Key1,Key2>& key)
        { return new Value(key.first, key.second); }
    };

    template <typename Value, typename Key1, typename Key2, typename Key3>
    struct LRUCacheHelper<Value,Tuple<Key1,Key2,Key3> >
    {
        static Value* NewValue(const Tuple<Key1,Key2,Key3>& key)
        { return new Value(key.first, key.second, key.third); }
    };

    template <typename Value, typename Key1, typename Key2, typename Key3, typename Key4>
    struct LRUCacheHelper<Value,Tuple<Key1,Key2,Key3,Key4> >
    {
        static Value* NewValue(const Tuple<Key1,Key2,Key3,Key4>& key)
        {
            return new Value(key.first, key.second, key.third, key.fourth);
        }
    };

    template <typename Value, typename Key1, typename Key2, typename Key3, typename Key4,
              typename Key5>
    struct LRUCacheHelper<Value,Tuple<Key1,Key2,Key3,Key4,Key5> >
    {
        static Value* NewValue(const Tuple<Key1,Key2,Key3,Key4,Key5>& key)
        {
            return new Value(key.first, key.second, key.third, key.fourth, key.fifth);
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
     * Special: if Key is a Tuple<Key1, Key2, ...> (up to 4), then value takes that many args:
     *
     *    Tuple<Key1,Key2> key(key1,key2);
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

        shared_ptr<Value> get(const Key& key)
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
                shared_ptr<Value> value(LRUCacheHelper<Value,Key>::NewValue(key));
                // Remove items from the cache as necessary.
                while (_entries.size() >= _nmax) {
                    _cache.erase(_entries.back().first);
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

        typedef std::pair<Key, shared_ptr<Value> > Entry;
        std::list<Entry> _entries;

        typedef typename std::list<Entry>::iterator ListIter;
        std::map<Key, ListIter> _cache;

        typedef typename std::map<Key, ListIter>::iterator MapIter;
    };

}

#endif
