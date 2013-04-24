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

#ifndef LRUCACHE_H
#define LRUCACHE_H

#include <list>
#include <map>

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
        LRUCache(size_t nmax) : _nmax(nmax) 
        {
            //std::cout<<"New LRUCache "<<this<<" with nmax = "<<_nmax<<std::endl;
        }

        /**
         * @brief Destructor
         *
         * Delete all items stored in the cache.
         */
        ~LRUCache() {
            //std::cout<<"Delete LRUCache "<<this<<std::endl;
            clear(); 
        }

        void clear() 
        { _cache.clear(); _entries.clear(); }

        boost::shared_ptr<Value> get(const Key& key)
        {
            //std::cout<<"LRUCache "<<this<<": get Key "<<&key<<std::endl;
            //std::cout<<"cache has "<<_cache.size()<<" items\n";
            //std::cout<<"entries has "<<_entries.size()<<" items\n";
            assert(_entries.size() == _cache.size());
            MapIter iter = _cache.find(key);
            if (iter != _cache.end()) {
                //std::cout<<"Found key in cache"<<std::endl;
                //std::cout<<"value = "<<iter->second->second.get()<<std::endl;
                // Item is cached.
                // Move it to the front of the list.
                if (iter != _cache.begin()) 
                    _entries.splice(_entries.begin(), _entries, iter->second);
                //std::cout<<"Moved key to front\n";
                // Return the item's value
                assert(_entries.size() == _cache.size());
                return iter->second->second;
            } else {
                //std::cout<<"key not in cache yet"<<std::endl;
                // Item is not cached.
                // Make a new one.
                boost::shared_ptr<Value> value(LRUCacheHelper<Value,Key>::NewValue(key));
                //std::cout<<"Made new value "<<value.get()<<std::endl;
                // Remove items from the cache as necessary.
                while (_entries.size() >= _nmax) {
                    //std::cout<<"Erasing element from back"<<std::endl;
                    bool erased = _cache.erase(_entries.back().first);
                    //std::cout<<"erased from _cache: "<<erased<<std::endl;
                    assert(erased);
                    _entries.pop_back();
                    //std::cout<<"erased from _entries"<<std::endl;
                }
                // Add the new value to the front.
                _entries.push_front(Entry(key,value));
                //std::cout<<"Added new entry to front of list"<<std::endl;
                // Also put it in the cache
                _cache[key] = _entries.begin();
                //std::cout<<"Added new entry to cache"<<std::endl;
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

#endif // LRUCACHE_H

