# $Id$
# Expect variables CXX and OPTFLAGS to be set in environment.
# Also need paths: CFITSIO_INCLUDE (if USE_IMAGES is set), FFTW_INCLUDE, 
# and TMV_INCLUDE (if USE_LAGUERRE is set)

SUBDIRS = 
SRC = SBProfile.cpp SBPixel.cpp SBParse.cpp Shear.cpp
OBJ = SBProfile.o SBPixel.o SBParse.o Shear.o
INCLUDES = -I ../utilities2 -I $(FFTW_INCLUDE)
CXXFLAGS = $(OPTFLAGS) $(INCLUDES)

ifdef USE_LAGUERRE
SRC += Laguerre.cpp
OBJ += Laguerre.o
CXXFLAGS += -DUSE_LAGUERRE
ifndef USE_TMV
USE_TMV = true
endif
endif

ifdef USE_TMV
INCLUDES += -I $(TMV_INCLUDE)
CXXFLAGS += -DUSE_TMV
export USE_TMV
endif

ifdef USE_IMAGES
INCLUDES += -I ../images
CXXFLAGS += -DUSE_IMAGES
endif

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
