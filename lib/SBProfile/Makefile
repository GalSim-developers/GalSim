# $Id$
# Makefile for SBProfile examples.  You'll probably need to set the following variables according
# to your setup:  FFTW_PATH, CFITSIO_PATH, TMV_PATH - and the *_INCLUDE, *_LIB if they don't
# just append /include and /lib to the *_PATH values.
# and perhaps edit the lines where TMV libraries are specified, depeding on your TMV setup.
# Also there are compilation 

CXX = g++-4
OPTFLAGS = -O2 -Wall -Werror -g

SUBDIRS = utilities2 sbprofile
INCLUDES = -I utilities2 -I sbprofile

CXXFLAGS = $(OPTFLAGS) $(INCLUDES)
LIBS = -lm 

# Variable to set to anything non-null if you want to use Gary's Image and FITSImage classes
# If it is not set, you currently have no way to produce images!
USE_IMAGES = true
# Variable to set if you want to compile with Laguerre functions:
USE_LAGUERRE = true
# Variable set to use the TMV library.  Right now SBPixel will fail without this set.
USE_TMV = true

# Info on where to find fftw:
FFTW_PATH = /sw
FFTW_INCLUDE = $(FFTW_PATH)/include
FFTW_LIB = $(FFTW_PATH)/lib
INCLUDES += -I $(FFTW_INCLUDE)
LIBS += -L$(FFTW_LIB) -lfftw3
export FFTW_INCLUDE

# and CFITSIO
CFITSIO_PATH = /sw
CFITSIO_INCLUDE = $(CFITSIO_PATH)/include
CFITSIO_LIB = $(CFITSIO_PATH)/lib
export CFITSIO_INCLUDE

# and TMV
TMV_PATH = /Users/Mike
TMV_INCLUDE = $(TMV_PATH)/include
TMV_LIB = $(TMV_PATH)/lib
export TMV_INCLUDE

SUBOBJ = utilities2/BinomFact.o images/FITS.o utilities2/Interpolant.o \
	utilities2/StringStuff.o utilities2/fft.o utilities2/Table.o utilities2/Pset.o \
	sbprofile/SBProfile.o sbprofile/SBParse.o sbprofile/SBPixel.o sbprofile/Shear.o

SRC = $(shell ls *.cpp)
OBJ = $(SUBOBJ)

ifdef USE_LAGUERRE
export USE_LAGUERRE
# need TMV if using Laguerre functions
ifndef USE_TMV
USE_TMV = true
endif
SUBOBJ += sbprofile/Laguerre.o
CXXFLAGS += -DUSE_LAGUERRE
INCLUDES += -I $(FFTW_INCLUDE)
endif

ifdef USE_IMAGES
export USE_IMAGES
SUBDIRS += images
SUBOBJ += images/Image.o images/FITSImage.o 
INCLUDES += -I images
CXXFLAGS += -DUSE_IMAGES
LIBS += -L$(CFITSIO_LIB) -lcfitsio 
endif

# Additional directives if TMV is being used
ifdef USE_TMV
export USE_TMV
INCLUDES += -I $(TMV_INCLUDE)
CXXFLAGS += -DUSE_TMV
TMVLINK := $(shell cat $(TMV_PATH)/share/tmv-link)
LIBS += -L$(TMV_LIB) -ltmv_symband $(TMVLINK)
endif

all: depend subs SBDraw Shera

SBDraw: SBDraw.o $(SUBOBJ)
	$(CXX) $(CXXFLAGS) $^  $(LIBS) -o $@
Shera: Shera.o $(SUBOBJ)
	$(CXX) $(CXXFLAGS) $^  $(LIBS) -o $@

###############################################################
## Standard stuff:
###############################################################

export OPTFLAGS
export CXX

subs:
	for dir in $(SUBDIRS); do (cd $$dir && $(MAKE)); done

depend:
	for dir in $(SUBDIRS); do (cd $$dir && $(MAKE) depend); done
	$(CXX) $(CXXFLAGS) -MM $(SRC) > .$@

clean:
	for dir in $(SUBDIRS); do (cd $$dir && $(MAKE) clean); done
	rm -f *.o *~ *.dvi *.aux core .depend

ifeq (.depend, $(wildcard .depend))
include .depend
endif

.PHONY: all install dist depend clean 
