/****************************************************************
  Copyright 2003, 2004 Christopher Hirata: original code
  2007, 2009, 2010 Rachel Mandelbaum: minor modifications

  For a copy of the license, see COPYING; for more information,
  including contact information, see README. 

  This file is part of the meas_shape distribution.

  Meas_shape is free software: you can redistribute it and/or modify it
  under the terms of the GNU General Public License as published by the
  Free Software Foundation, either version 3 of the License, or (at your
  option) any later version.

  Meas_shape is distributed in the hope that it will be useful, but
  WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with meas_shape.  If not, see <http://www.gnu.org/licenses/>.
 *******************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "fitsio.h"
#include "GalSim.h"

#define Pi 3.141592653589793
#define ARCSEC 2.53 /* pixels */

/* Code to convert 2D integer images from FITS to TEXT */

/* MAXIMUM NUMBER OF PIXELS */
#define MPIX 1000000

/* Arguments: argv[1] = input file, argv[2] = output file */

static int read_image(char FileName[], galsim::hsm::RectImage *MyImage) 
{

    fitsfile *fptr;
    int bitpix;
    long naxes[2];
    int naxis;
    int status=0;
    long fpixel[2];
    int anynul=0;
    long i,j;
    long nelem;

    unsigned short int *im_usht=0;
    int                *im_int=0;
    long               *im_lng=0;
    float              *im_flt=0;
    double             *im_dbl=0;

    fpixel[0]=fpixel[1]=1;

    fits_open_file(&fptr,FileName,READONLY,&status);
    if(status) {printf("Error %d\n",status); exit(1);}

    fits_get_img_param(fptr,2,&bitpix,&naxis,naxes,&status);
    if(status) {printf("Error %d\n",status); exit(1);}
    if (naxis!=2) {printf("Error: this program is designed to work with two axes.\n"); exit(1);}

    nelem=naxes[0]*naxes[1];

    allocate_rect_image(MyImage, 0, naxes[0]-1, 0, naxes[1]-1);

    switch(bitpix) {
      case BYTE_IMG:
           im_usht = (unsigned short int *) malloc((size_t) (nelem*sizeof(unsigned short int)));
           fits_read_img_usht(fptr,0,1,nelem,0,im_usht,&anynul,&status);
           break;
      case SHORT_IMG:
           im_int = (int *) malloc((size_t) (nelem*sizeof(int)));
           fits_read_img_int(fptr,0,1,nelem,0,im_int,&anynul,&status);
           break;
      case LONG_IMG:
           im_lng = (long *) malloc((size_t) (nelem*sizeof(long)));
           fits_read_img_lng(fptr,0,1,nelem,0,im_lng,&anynul,&status);
           break;
      case FLOAT_IMG:
           im_flt = (float *) malloc((size_t) (nelem*sizeof(float)));
           fits_read_img_flt(fptr,0,1,nelem,0,im_flt,&anynul,&status);
           break;
      case DOUBLE_IMG:
           im_dbl = (double *) malloc((size_t) (nelem*sizeof(double)));
           fits_read_img_dbl(fptr,0,1,nelem,0,im_dbl,&anynul,&status);
           break;
      default:
           printf("Error: I don't recognize the data type.\n");
           exit(1);
    }
    if(status) {printf("Error %d\n",status); exit(1);}

    for(i=0;i<naxes[1];i++) {
        for(j=0;j<naxes[0];j++) {
            switch(bitpix) {
              case BYTE_IMG:
                   MyImage->image[j][i] = (double) im_usht[j+i*naxes[0]];
                   break;
              case SHORT_IMG:
                   MyImage->image[j][i] = (double) im_int[j+i*naxes[0]];
                   break;
              case LONG_IMG:
                   MyImage->image[j][i] = (double) im_lng[j+i*naxes[0]];
                   break;
              case FLOAT_IMG:
                   MyImage->image[j][i] = (double) im_flt[j+i*naxes[0]];
                   break;
              case DOUBLE_IMG:
                   MyImage->image[j][i] = im_dbl[j+i*naxes[0]];
                   break;
              default:
                   printf("Error: I don't recognize the data type.\n");
                   exit(1);
            }
        }}

    switch(bitpix) {
      case BYTE_IMG:
           free((char *)im_usht);
           break;
      case SHORT_IMG:
           free((char *)im_int);
           break;
      case LONG_IMG:
           free((char *)im_lng);
           break;
      case FLOAT_IMG:
           free((char *)im_flt);
           break;
      case DOUBLE_IMG:
           free((char *)im_dbl);
           break;
      default:
           printf("Error: I don't recognize the data type.\n");
           exit(1);
    }

    fits_close_file(fptr,&status);
    return(status);

}

/* Arguments:
 * argv[1] = galaxy image image (FITS) file
 * argv[2] = PSF image (FITS) file
 * argv[3] = centroid x guess
 * argv[4] = centroid y guess
 * argv[5] = sky variance per pixel
 * argv[6] = estimator
 * argv[7] = sky level in galaxy and PSf postage stamps, including soft bias
 */

int main(int argc, char **argv) 
{

    galsim::hsm::RectImage AtlasImage, PSFImage;
    int status;
    long i,j;
    galsim::hsm::ObjectData GalaxyData, PSFData;
    //double x00, y00;
    double shearsig, skyvar,FLUX_OFFSET;

    /* Check the number of arguments */
    if (argc!=8) {
        fprintf(
            stderr,
            "Usage:\n\t%s gal_image_file PSF_image_file gal_x_centroid_guess "
            "gal_y_centroid_guess skyvar shear_estimator sky\n",argv[0]);
        exit(1);
    }

    /* Get sky variance */
    sscanf(argv[5], "%lf", &skyvar);

    /* get sky level, including soft bias, in galaxy and PSF images */
    sscanf(argv[7],"%lf",&FLUX_OFFSET);

    /* Read atlas images, initialize their data */
    status = read_image(argv[1], &AtlasImage);
    if (status) {
        fprintf(stderr,"Error %d in reading atlas image from file %s.\n", status,argv[1]);
        exit(status);
    }
    for(i=AtlasImage.xmin;i<=AtlasImage.xmax;i++) {
        for(j=AtlasImage.ymin;j<=AtlasImage.ymax;j++) {
            AtlasImage.image[i][j] -= FLUX_OFFSET;
        }
    }
    sscanf(argv[3], "%lf", &(GalaxyData.x0));
    sscanf(argv[4], "%lf", &(GalaxyData.y0));
    GalaxyData.sigma = 2.5 * ARCSEC;
    //x00 = GalaxyData.x0;
    //y00 = GalaxyData.y0;

    /* Read PSF and initialize */
    status = read_image(argv[2], &PSFImage);
    if (status) {
        fprintf(stderr,"Error %d in reading PSF image from file %s.\n", status,argv[2]);
        exit(status);
    }
    for(i=PSFImage.xmin;i<=PSFImage.xmax;i++) {
        for(j=PSFImage.ymin;j<=PSFImage.ymax;j++) {
            PSFImage.image[i][j] -= FLUX_OFFSET;
        }
    }

    PSFData.x0 = 0.5 * (PSFImage.xmin + PSFImage.xmax);
    PSFData.y0 = 0.5 * (PSFImage.ymin + PSFImage.ymax);
    PSFData.sigma = 1.0 * ARCSEC;

    status = (int) general_shear_estimator(
        &AtlasImage, &PSFImage, &GalaxyData, &PSFData, argv[6], 0xe);

    /* Compute very rough error estimate */
    shearsig = sqrt( 4. * Pi * skyvar ) * 
        GalaxyData.sigma / (GalaxyData.resolution * GalaxyData.flux);

    printf("%d %13.9lf %13.9lf %c %13.9lf %13.9f %13.9f %13.9f %13.9f\n",
           status, GalaxyData.e1, GalaxyData.e2,GalaxyData.meas_type,
           GalaxyData.responsivity, GalaxyData.resolution, shearsig,
           GalaxyData.sigma, GalaxyData.flux);

    deallocate_rect_image(&AtlasImage);
    deallocate_rect_image(&PSFImage);

#if 1
    if (GalaxyData.resolution <= 0.25) return 81;
    if (GalaxyData.e1*GalaxyData.e1 + GalaxyData.e2*GalaxyData.e2 >= 100.0) return 80;
#endif

    if (status) fprintf(stderr, "Error #%d: ", status);
    return (status);
}
