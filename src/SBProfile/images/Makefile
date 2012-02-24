#   $Id: Makefile,v 1.7 2011/02/14 15:22:13 garyb Exp $ 
# Assume that CXX and OPTFLAGS variables have been obtained from environment
# and CFITSIO_INCLUDE directory needed too.
SUBDIRS = 
SRC = $(shell ls *.cpp)
OBJ = FITS.o Image.o FITSImage.o HeaderFromStream.o
INCLUDES = -I ../utilities2 -I $(CFITSIO_INCLUDE)
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
