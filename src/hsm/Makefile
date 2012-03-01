meas_shape: meas_shape.c psfcorr.c psfcorr.h
	gcc meas_shape.c psfcorr.c -o meas_shape.x -lm -Wall -lcfitsio -DREGAUSS_TOO_SMALL

meas_moments: meas_moments.c psfcorr.c psfcorr.h
	gcc meas_moments.c psfcorr.c -o meas_moments.x -lm -Wall -lcfitsio -DREGAUSS_TOO_SMALL

#meas_shape: meas_shape.c psfcorr.c psfcorr.h
#	gcc meas_shape.c psfcorr.c -o meas_shape.x -lm -Wall -L/u/rmandelb/software/cfitsio -lcfitsio -DREGAUSS_TOO_SMALL
