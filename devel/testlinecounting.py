# Copyright (c) 2012-2018 by the GalSim developers team on GitHub
# https://github.com/GalSim-developers
#
# This file is part of GalSim: The modular galaxy image simulation toolkit.
# https://github.com/GalSim-developers/GalSim
#
# GalSim is free software: redistribution and use in source and binary forms,
# with or without modification, are permitted provided that the following
# conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions, and the disclaimer given in the accompanying LICENSE
#    file.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions, and the disclaimer given in the documentation
#    and/or other materials provided with the distribution.
#

"""Totally dumb thing Melanie Simet put together to test line-counting algorithms in Python.
"""

import numpy
import time

ntrials=10

def bufcount(filename):
    f = open(filename)                  
    lines = 0
    buf_size = 1024 * 1024
    read_f = f.read # loop optimization

    buf = read_f(buf_size)
    while buf:
        # Note: Melanie points out that this next line isn't exactly correct...
        # There's a slight chance the buffer test could undercount lines: it tests for the first 
        # character being # and for the pattern \n#, but if the buffer happened to break in the 
        # middle of a line where there was a random hash, it will mistakenly decrement the count.
        # So be aware of that if you're thinking of using it.
        if buf[0]=='#':
            lines -= 1
        lines += buf.count('\n')
        lines -= buf.count("\n#")
        buf = read_f(buf_size)
    f.close()
    return lines

def numpycount(filename):
    f = numpy.loadtxt(filename)
    return f.shape[0]
    
def dumbcount(filename):
    f = open(filename)
    nr_of_lines = sum(1 for line in f if not line.startswith('#'))
    f.close()
    return nr_of_lines

def dumbcount_v2(filename):
    f = open(filename)
    nr_of_lines = sum(1 for line in f if not line[0] == '#')
    f.close()
    return nr_of_lines

def dumbcount_v3(filename):
    f = open(filename)
    nr_of_lines = len([line for line in f if not line[0] == '#'])
    f.close()
    return nr_of_lines

def readlinescount(filename):
    f = open(filename)
    lines = f.readlines()
    oklines = [ lines for line in lines if line[0] != '#' ]
    nobj = len(oklines)
    f.close()
    return nobj

def readlines2count(filename):
    f = open(filename)
    lines = f.readlines()
    nobj = sum(1 for line in lines if line[0] != '#')
    f.close()
    return nobj

def main():
    filename = '../tests/lensing_reference_data/tmp.txt'
    filename_with_hashes = '../tests/lensing_reference_data/nfw_lens.dat'
    
    numpytime=0.
    numpyhashtime=0.
    dumbtime=0.
    dumbhashtime=0.
    dumbtimev2=0.
    dumbhashtimev2=0.
    dumbtimev3=0.
    dumbhashtimev3=0.
    buftime=0.
    bufhashtime=0.
    readlinestime=0.
    readlineshashtime=0.
    readlines2time=0.
    readlines2hashtime=0.
    
    for i in range(ntrials):
        t1=time.time()
        n = dumbcount(filename)
        t2=time.time()
        print "Dumb test: trial without comments", i, "took", t2-t1, "with result", n, "(should be 10000)"
        dumbtime+=t2-t1
    for i in range(ntrials):
        t1=time.time()
        n = dumbcount(filename_with_hashes)
        t2=time.time()
        print "Dumb test: trial with comments", i, "took", t2-t1, "with result", n, "(should be 599)"
        dumbhashtime+=t2-t1

    for i in range(ntrials):
        t1=time.time()
        n = dumbcount_v2(filename)
        t2=time.time()
        print "Dumb test v2: trial without comments", i, "took", t2-t1, "with result", n, "(should be 10000)"
        dumbtimev2+=t2-t1
    for i in range(ntrials):
        t1=time.time()
        n = dumbcount_v2(filename_with_hashes)
        t2=time.time()
        print "Dumb test v2: trial with comments", i, "took", t2-t1, "with result", n, "(should be 599)"
        dumbhashtimev2+=t2-t1
    
    for i in range(ntrials):
        t1=time.time()
        n = dumbcount_v3(filename)
        t2=time.time()
        print "Dumb test v3: trial without comments", i, "took", t2-t1, "with result", n, "(should be 10000)"
        dumbtimev3+=t2-t1
    for i in range(ntrials):
        t1=time.time()
        n = dumbcount_v3(filename_with_hashes)
        t2=time.time()
        print "Dumb test v3: trial with comments", i, "took", t2-t1, "with result", n, "(should be 599)"
        dumbhashtimev3+=t2-t1
    
    for i in range(ntrials):
        t1=time.time()
        n = bufcount(filename)
        t2=time.time()
        print "Buffer test: trial without comments", i, "took", t2-t1, "with result", n, "(should be 10000)"
        buftime+=t2-t1
    for i in range(ntrials):
        t1=time.time()
        n = bufcount(filename_with_hashes)
        t2=time.time()
        print "Buffer test: trial with comments", i, "took", t2-t1, "with result", n, "(should be 599)"
        bufhashtime+=t2-t1
    
    for i in range(ntrials):
        t1=time.time()
        n = numpycount(filename)
        t2=time.time()
        print "Numpy test: trial without comments", i, "took", t2-t1, "with result", n, "(should be 10000)"
        numpytime+=t2-t1
    for i in range(ntrials):
        t1=time.time()
        n = numpycount(filename_with_hashes)
        t2=time.time()
        print "Numpy test: trial with comments", i, "took", t2-t1, "with result", n, "(should be 599)"
        numpyhashtime+=t2-t1
    
    for i in range(ntrials):
        t1=time.time()
        n = readlinescount(filename)
        t2=time.time()
        print "Readlines test: trial without comments", i, "took", t2-t1, "with result", n, "(should be 10000)"
        readlinestime+=t2-t1
    for i in range(ntrials):
        t1=time.time()
        n = readlinescount(filename_with_hashes)
        t2=time.time()
        print "Readlines test: trial with comments", i, "took", t2-t1, "with result", n, "(should be 599)"
        readlineshashtime+=t2-t1
    
    for i in range(ntrials):
        t1=time.time()
        n = readlines2count(filename)
        t2=time.time()
        print "Readlines2 test: trial without comments", i, "took", t2-t1, "with result", n, "(should be 10000)"
        readlines2time+=t2-t1
    for i in range(ntrials):
        t1=time.time()
        n = readlines2count(filename_with_hashes)
        t2=time.time()
        print "Readlines2 test: trial with comments", i, "took", t2-t1, "with result", n, "(should be 599)"
        readlines2hashtime+=t2-t1
    
    print "***FINAL RESULTS***"
    print "Dumb test:", dumbtime/ntrials, "without comments"
    print "Dumb test:", dumbhashtime/ntrials, "with comments"
    print "Dumb test v2:", dumbtimev2/ntrials, "without comments"
    print "Dumb test v2:", dumbhashtimev2/ntrials, "with comments"
    print "Dumb test v3:", dumbtimev3/ntrials, "without comments"
    print "Dumb test v3:", dumbhashtimev3/ntrials, "with comments"
    print "Buffer test:", buftime/ntrials, "without comments"
    print "Buffer test:", bufhashtime/ntrials, "with comments"
    print "Numpy test:", numpytime/ntrials, "without comments"
    print "Numpy test:", numpyhashtime/ntrials, "with comments"
    print "Readlines test:", readlinestime/ntrials, "without comments"
    print "Readlines test:", readlineshashtime/ntrials, "with comments"
    print "Readlines2 test:", readlines2time/ntrials, "without comments"
    print "Readlines2 test:", readlines2hashtime/ntrials, "with comments"
    
if __name__=='__main__':
    main()
