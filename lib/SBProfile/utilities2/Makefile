#   $Id: Makefile,v 1.3 2012/01/01 18:56:10 garyb Exp $ 
# Expect variables CXX and OPTFLAGS to be set in environment
# and FFTW_INCLUDE path
SUBDIRS = 
SRC = $(shell ls *.cpp)
OBJ = BinomFact.o fft.o StringStuff.o Poisson.o Table.o Pset.o Interpolant.o
INCLUDES = -I $(FFTW_INCLUDE)
CXXFLAGS = $(OPTFLAGS) $(INCLUDES)

all: depend $(OBJ)

###############################################################
## Standard stuff:
###############################################################

export OPTFLAGS
export CXX

depend:
	for dir in $(SUBDIRS); do (cd $$dir && $(MAKE) depend); done
	$(CXX) $(CXXFLAGS) -MM $(SRC) > .$@

clean:
	for dir in $(SUBDIRS); do (cd $$dir && $(MAKE) clean); done
	rm -f *.o *~ core .depend

ifeq (.depend, $(wildcard .depend))
include .depend
endif

.PHONY: all install dist depend clean 
