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

// ---------------------------------------------------------------------------------------------------------------------------------
// Copyright 2000, Paul Nettle. All rights reserved.
//
// You are free to use this source code in any commercial or non-commercial product.
//
// mmgr.h - Memory manager & tracking software
//
// The most recent version of this software can be found at: ftp://ftp.GraphicsPapers.com/pub/ProgrammingTools/MemoryManagers/
//
// [NOTE: Best when viewed with 8-character tabs]
// ---------------------------------------------------------------------------------------------------------------------------------

#ifndef GalSim_MMGR_H
#define GalSim_MMGR_H

// ---------------------------------------------------------------------------------------------------------------------------------
// For systems that don't have the __FUNCTION__ variable, we can just define it here
// ---------------------------------------------------------------------------------------------------------------------------------

//#define __FUNCTION__ __func__

// ---------------------------------------------------------------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------------------------------------------------------------

typedef struct tag_au
{
    size_t actualSize;
    size_t reportedSize;
    void  *actualAddress;
    void  *reportedAddress;
    char  sourceFile[40];
    char  sourceFunc[40];
    size_t sourceLine;
    size_t allocationType;
    bool  breakOnDealloc;
    bool  breakOnRealloc;
    size_t allocationNumber;
    struct tag_au *next;
    struct tag_au *prev;
} sAllocUnit;

typedef struct
{
    size_t totalReportedMemory;
    size_t totalActualMemory;
    size_t peakReportedMemory;
    size_t peakActualMemory;
    size_t accumulatedReportedMemory;
    size_t accumulatedActualMemory;
    size_t accumulatedAllocUnitCount;
    size_t totalAllocUnitCount;
    size_t peakAllocUnitCount;
} sMStats;

// ---------------------------------------------------------------------------------------------------------------------------------
// External constants
// ---------------------------------------------------------------------------------------------------------------------------------

extern const size_t m_alloc_unknown;
extern const size_t m_alloc_new;
extern const size_t m_alloc_new_array;
extern const size_t m_alloc_malloc;
extern const size_t m_alloc_calloc;
extern const size_t m_alloc_realloc;
extern const size_t m_alloc_delete;
extern const size_t m_alloc_delete_array;
extern const size_t m_alloc_free;

// ---------------------------------------------------------------------------------------------------------------------------------
// Used by the macros
// ---------------------------------------------------------------------------------------------------------------------------------

void m_setOwner(const char *file, const size_t line, const char *func);

// ---------------------------------------------------------------------------------------------------------------------------------
// Allocation breakpoints
// ---------------------------------------------------------------------------------------------------------------------------------

bool &m_breakOnRealloc(void *reportedAddress);
bool &m_breakOnDealloc(void *reportedAddress);

// ---------------------------------------------------------------------------------------------------------------------------------
// The meat of the memory tracking software
// ---------------------------------------------------------------------------------------------------------------------------------

void *m_allocator(
    const char *sourceFile, const size_t sourceLine, const char *sourceFunc,
    const size_t allocationType, const size_t reportedSize);
void *m_reallocator(
    const char *sourceFile, const size_t sourceLine, const char *sourceFunc,
    const size_t reallocationType, const size_t reportedSize, void *reportedAddress);
void  m_deallocator(
    const char *sourceFile, const size_t sourceLine, const char *sourceFunc,
    const size_t deallocationType, const void *reportedAddress);

// ---------------------------------------------------------------------------------------------------------------------------------
// Utilitarian functions
// ---------------------------------------------------------------------------------------------------------------------------------

bool m_validateAddress(const void *reportedAddress);
bool m_validateAllocUnit(const sAllocUnit *allocUnit);
bool m_validateAllAllocUnits();

// ---------------------------------------------------------------------------------------------------------------------------------
// Unused RAM calculations
// ---------------------------------------------------------------------------------------------------------------------------------

size_t m_calcUnused(const sAllocUnit *allocUnit);
size_t m_calcAllUnused();

// ---------------------------------------------------------------------------------------------------------------------------------
// Logging and reporting
// ---------------------------------------------------------------------------------------------------------------------------------

void m_dumpAllocUnit(const sAllocUnit *allocUnit, const char *prefix = "");
void m_dumpMemoryReport(const char *filename = "memreport.log", const bool overwrite = true);
sMStats m_getMemoryStatistics();

// ---------------------------------------------------------------------------------------------------------------------------------
// Variations of global operators new & delete
// ---------------------------------------------------------------------------------------------------------------------------------

void *operator new(size_t reportedSize) throw(std::bad_alloc);
void *operator new[](size_t reportedSize) throw(std::bad_alloc);
void *operator new(size_t reportedSize, const char *sourceFile, int sourceLine) throw(std::bad_alloc);
void *operator new[](size_t reportedSize, const char *sourceFile, int sourceLine) throw(std::bad_alloc);
void operator delete(void *reportedAddress) throw();
void operator delete[](void *reportedAddress) throw();

#endif // _H_MMGR

// ---------------------------------------------------------------------------------------------------------------------------------
// Macros -- "Kids, please don't try this at home. We're trained professionals here." :)
// ---------------------------------------------------------------------------------------------------------------------------------

#define new (m_setOwner (__FILE__,__LINE__,__FUNCTION__),false) ? NULL : new
#define delete (m_setOwner (__FILE__,__LINE__,__FUNCTION__),false) ? m_setOwner("",0,"") : delete
#define malloc(sz) m_allocator (__FILE__,__LINE__,__FUNCTION__,m_alloc_malloc,sz)
#define calloc(sz) m_allocator (__FILE__,__LINE__,__FUNCTION__,m_alloc_calloc,sz)
#define realloc(ptr,sz) m_reallocator(__FILE__,__LINE__,__FUNCTION__,m_alloc_realloc,sz,ptr)
#define free(ptr) m_deallocator(__FILE__,__LINE__,__FUNCTION__,m_alloc_free,ptr)

// ---------------------------------------------------------------------------------------------------------------------------------
// mmgr.h - End of file
// ---------------------------------------------------------------------------------------------------------------------------------
