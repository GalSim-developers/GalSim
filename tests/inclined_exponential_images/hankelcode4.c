/* -*- c++ -*-
 * Copyright (c) 2012-2018 by the GalSim developers team on GitHub
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

// Testbed code for fast generation of galaxy model surface brightness distributions
//
// Lance Miller June/July/August 2016
// Bryan Gillis August/December 2016

// library dependencies:
//     cfitsio
//     fftw3

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <complex.h>
#include <fftw3.h>
#include <fitsio.h>

// set the image domain oversampling - must match the PSF oversampling
int oversampling = 5;

int makecirculargalaxy(double sersic_n, double rfiducial, double trunc_factor, int modeldim, double *f)
{
  // make an oversampled galaxy model
  double xx, yy, r;
  int cen, x, y, os, p, ox, oy, xs, ys;

  // galaxy parameters
  cen = modeldim/2;

  // image domain oversampling (only for this model generation) - not the same as oversampling in the rest of the code
  os = 5;

  for (y=1; y<modeldim; y++)
    {
      // swap quadrants to ensure model is even and has a real FT
      ys = y - modeldim/2;
      if (ys < 0) ys += modeldim;
      for (x=1; x<modeldim; x++)
        {
          // swap quadrants to ensure model is even and has a real FT
          xs = x - modeldim/2;
          if (xs < 0) xs += modeldim;
          p = ys*modeldim+xs;
          f[p] = 0.;
          // make an oversampled distribution
          for (oy=0; oy<os; oy++)
            {
              yy = y + (0.5+(double)oy)/(double)os - 0.5 - cen;
              for (ox=0; ox<os; ox++)
                {
                  xx = x + (0.5+(double)ox)/(double)os - 0.5 - cen;
                  r = sqrt( pow(yy,2) + pow(xx,2) )/rfiducial;
                  if (r < trunc_factor)
                    {
                      r = pow( r, (1./sersic_n) );
                      f[p] += exp(-r)/(os*os);
                    }
                }
            }
        }
    }

  return(0);

}


int write2Dfits(char *fname, double *f, int pwidth, int pheight)
{
  /*  write out image as a fits file */

  fitsfile *afptr;
  int status = 0;
  int anaxis;
  long anaxes[2], fpixel[2]={1,1};
  int bitpix;
  int size;

  bitpix = -32;
  anaxis = 2;
  anaxes[0] = pwidth;
  anaxes[1] = pheight;

  size = anaxes[0]*anaxes[1];

  fits_create_file(&afptr, fname, &status);
  fits_create_img(afptr, bitpix, anaxis, anaxes, &status);

  if (status) {
    fits_report_error(stderr, status); /* print error message */
    return(status);
  }

  /* write all data into image array */
  if (fits_write_pix(afptr, TDOUBLE, fpixel, size, f, &status) )
    {
      printf(" error reading pixel data \n");
      exit (2);
    }

  /* close the fits file */

  fits_close_file(afptr, &status);

  if (status) {
    fits_report_error(stderr, status); /* print error message */
    exit(2);
  }

  return 0;

}


void alias(int dim, int odim, fftw_complex *in, fftw_complex *out)
{
  // function to alias large 2D hermitian array into a smaller 2D (non-hermitian) array
  // assumes FFTW convention for hermitian array storage

  int x, y, cen, ocen, xx, yy, offset, ip, op;
  int hdim;
  double norm;

  // define coordinate in input FT that contains zero frequency component
  cen = 0;
  // define coordinate in output FT that contains zero frequency component
  ocen = 0;

  // define a coordinate offset so that values always stay positive
  // but which is a multiple of the output dimension
  // such that the defined zero-frequency component of the input array
  // is mapped onto the defined zero-frequency component of the output array
  offset = odim*(1 + (dim/odim)) + ocen - cen;

  // initialise output to zero
  for (op=0; op<(odim*odim); op++)
    {
      out[op] = 0.;
    }

  // define halfwidth of hermitian half arrays along x-axis
  hdim = 1 + dim/2;

  // equalise weight of the redundant elements
  for (y=0; y<dim; y++)
    {
      x = hdim-1;
      in[y*hdim+x] /= 2.;
      x = 0;
      in[y*hdim+x] /= 2.;
    }

  // loop through FT array and alias-downsample
  for (y=0; y<dim; y++)
    {
      // calculate coordinate in downsampled array
      yy = (y + offset) % odim;
      //printf (" %d %d \n",y, yy);
      for (x=0; x<hdim; x++)
        {
          // calculate coordinate in downsampled array
          xx = (x + offset) % odim;
          ip = y*hdim + x;
          op  = yy*odim + xx;
          out[op] += in[ip];
        }
    }

  // renormalise
  norm = creal(out[0]);
  for (op=0; op<odim*odim; op++)
    {
      out[op] /= norm;
    }


}


void thickdiskfunc(int dim, double e1, double e2, double scaleheight, float *thickdisk)
{
  /*
    calculates the Fourier transform for a sech^2 thick disk

    inputs:
     dim      int 1D dimension of 2D array
     e1, e2   input pseudo-ellipticity values - the e values that would be obtained for a
              thin disk at this ellipticity and position angle
     scaleheight   scaleheight of the thick disk sech^2 distribution

   outputs:
     thickdisk 2D float array holding the Fourier transform of the thick disk function
  */

  double posangle, cosangle, sinangle, emod, sini, factor, z, arg, val;
  int x, y, yy, hdim, p;

  // get position angle of minor axis of galaxy
  posangle = atan2(-e2,-e1)/2.;
  cosangle = cos(posangle);
  sinangle = sin(posangle);

  // get sin of inclination angle from e1,e2 values assuming these are for thin disk
  // (i.e. input parameters are "pseudo-ellipticity" - the values of ellipticity that
  // would be obtained if the galaxy really were a thin disk at the assumed inclination
  // we could of course directly input the inclination and position angle as the
  // model parameters instead
  emod = sqrt(e1*e1+e2*e2);
  sini = 2.*sqrt(emod)/(1.+emod);

  // argument scaling prefactor
  factor = pow(M_PI,2)*sini*scaleheight/dim;

  hdim = 1+dim/2;

  // work through elements of the 2D array to calculate the Fourier transform.
  // As it stands this is slow because of the exponential calculations.  It could
  // be greatly speed up by calculating a 1D lookup table of values and interpolating
  for (y=0; y<dim; y++)
    {
      yy = y<dim/2 ? y : y-dim;
      for (x=0; x<hdim; x++)
        {
          p = y*hdim + x;
          // coordinate projected onto minor axis
          z = x*cosangle + yy*sinangle;
          arg = fabs(factor*z);
          if (arg > 1.e-5)
            {
              // normal case where argument is not close to zero
              val = 2.*arg/(exp(arg)-exp(-arg));
            }
          else
            {
              // deal with case when arg -> 0
              val = 1./(1.+arg*arg/6.);
            }
          thickdisk[p] = val;
        }
    }

}

double rinterp(float *array, double r)
{
  // linear interpolation function

  int p,p1;
  double dr,f1,f2,val;

  p = (int)r;
  p1 = p+1;
  dr = r-p;

  f1 = array[p];
  f2 = array[p1];

  val = f1 + dr*(f2-f1);

  return(val);
}


int makehankel(fftw_complex *modelft, int hdim, float *hankelmodel)
{
  int x;

  for (x=0; x<hdim; x++)
    {
      hankelmodel[x] = creal(modelft[x]);
    }

  return (0);
}


int hankelresample (float *rmodelft, int mdim, double e1, double e2, double galsize, double rfiducial, int idim, float *rconvmodelft)
{
  // sample the Hankel transform model in scaled, sheared coordinates to generate FT of elliptical model

  // this version has no wrap-around effects included - makes models with high frequency
  // structure but is more correct for band-limited PSFs

  int x, y, yy, op, hdim, hmdim;
  double emod, rscaled, xp, yp, r;
  double a, b, c, d;

  // Nyquist dimensions
  hdim = 1 + idim/2;
  hmdim = 1 + mdim/2;

  emod = sqrt(e1*e1+e2*e2);

  // scale the galaxy size relative to the fiducial size in the circular model and define as major axis
  rscaled = galsize*mdim/(1.+emod)/rfiducial/idim;

  // shear transformation
  a = (1.+e1)*rscaled;
  b = c = e2*rscaled;
  d = (1.-e1)*rscaled;

  // sample the circular model FT to make the FT of a model with the right ellipticity and size
  for (y=0; y<idim; y++)
    {
      // treat top half of array as having negative frequencies
      yy = y<idim/2 ? y : y-idim;
      for (x=0; x<hdim; x++)
        {
          // output pixel element
          op = y*hdim + x;
          // transformed coordinates
          xp = a*x + b*yy;
          yp = c*x + d*yy;
          // radius
          r = sqrt(xp*xp + yp*yp);
          // interpolate
          if (r < hmdim-1)
            {
              rconvmodelft[op] = rinterp(rmodelft,r);
            }
          else
            {
              rconvmodelft[op] = 0.;
            }
        }
    }

  return(0);

}



int createshiftft(double xpos, double ypos, int dim, fftw_complex *xshiftft, fftw_complex *yshiftft)
{
  int p, pp, hdim;
  double arg;

  hdim = 1 + dim/2;

  for (p=0; p<hdim; p++)
    {
      arg = xpos*M_PI*(double)p/(double)(hdim-1);
      xshiftft[p] = cos(arg) - I*sin(arg);
    }

  for (p=0; p<dim; p++)
    {
      pp = p<dim/2 ? p : p-dim;
      arg = ypos*M_PI*(double)pp/(double)(hdim-1);
      yshiftft[p] = cos(arg) - I*sin(arg);
    }

  return(0);
}


int convolve(int dim, double scaleheight, fftw_complex *convmodelft, float *rconvmodelft, fftw_complex *PSFft, float *thickdiskft, fftw_complex *xshiftft, fftw_complex *yshiftft)
{
  // Fourier domain operations to cause image-domain convolution
  // multiply all the Fourier transforms together

  fftw_complex fs, ys;
  int x, y, p, hdim;

  hdim = 1 + dim/2;

  if (scaleheight > 0.)
    {
      //printf(" thick disk convolution %f \n",scaleheight);
      // convolution including a thick disk
      for (y=0; y<dim; y++)
        {
          ys = yshiftft[y];
          for (x=0; x<hdim; x++)
            {
              p = y*hdim + x;
              fs = PSFft[p]*thickdiskft[p]*xshiftft[x]*ys;
              convmodelft[p] = rconvmodelft[p]*fs;
            }
        }
    }
  else
    {
      // convolution without a thick disk
      for (y=0; y<dim; y++)
        {
          ys = yshiftft[y];
          for (x=0; x<hdim; x++)
            {
              p = y*hdim + x;
              fs = PSFft[p]*xshiftft[x]*ys;
              convmodelft[p] = rconvmodelft[p]*fs;
            }
        }
    }

  return (0);

}

void GenerateModel(
                   // parameters for output model galaxy
                   double e1,        // e1
                   double e2,        // e2
                   double galsize,   // scalelength (oversampled pixels)
                   double scaleheight, // scaleheight (oversampled pixels)
                   double xpos,      // x position shift (oversampled pixels)
                   double ypos,      // y position shift (oversampled pixels)
                   // nominal (fiducial) radius of input galaxy
                   double rfiducial,
                   // 1D dimension of output galaxy image
                   int odim,
                   // 1D dimension of intermediate resampled image
                   int idim,
                   // 1D dimension of large input circular galaxy image
                   int mdim,
                   // pointer to FFT array of input circular galaxy
                   float* rmodelft,
                   // pointer to FT array of resampled galaxy
                   float* resampledmodelft,
                   // pointer to PSF FT
                   fftw_complex* PSFft,
                   // pointer to thick disk FT
                   float* thickdiskft,
                   // pointer to x-shift FT,
                   fftw_complex* xshiftft,
                   // pointer to y-shift FT,
                   fftw_complex* yshiftft,
                   // pointer to convolved, oversampled model FT
                   fftw_complex* convmodelft,
                   // pointer to final downsampled, convolved FT
                   fftw_complex* dsmodelft
                   )
{
  // function to resample the input galaxy FT to create an output galaxy of specified size and ellipticity,

  // resample the Hankel transform
  hankelresample(rmodelft, mdim, e1, e2, galsize, rfiducial, idim, resampledmodelft);

  // create FT of 1D shifts in x and y
  createshiftft(xpos, ypos, idim, xshiftft, yshiftft);

  // create FT of thick disk
  if (scaleheight > 0.)
    {
      //printf(" thick disk with oversampled scaleheight %f \n",scaleheight); fflush(stdout);
      thickdiskfunc(idim, e1, e2, scaleheight, thickdiskft);
    }

  // convolve
  convolve(idim, scaleheight, convmodelft, resampledmodelft, PSFft, thickdiskft, xshiftft, yshiftft);

  /*
  int pp,x,y;
  int ihdim = 1 + idim/2;
  double* rsimage = (double*)calloc(idim*ihdim, sizeof(double));

  for (y=0; y<idim; y++)
    {
      for (x=0; x<ihdim; x++)
        {
          pp = y*ihdim + x;
          rsimage[pp] = thickdiskft[pp];
        }
    }
  remove("thickdiskft.fits");
  write2Dfits("thickdiskft.fits",rsimage,ihdim,idim);

  for (y=0; y<idim; y++)
    {
      for (x=0; x<ihdim; x++)
        {
          pp = y*ihdim + x;
          rsimage[pp] = cabs(convmodelft[pp]);
        }
    }
  remove("rsgalaxyft.fits");
  write2Dfits("rsgalaxyft.fits",rsimage,ihdim,idim);
  free(rsimage);
  */

  // call alias function
  alias(idim, odim, convmodelft, dsmodelft );

}

int main(int argc, char * argv[])
{
  // main function to set up the components.  All of this work would be done in Python apart from
  // the GenerateModel function

  // Check we have enough cline-args
  if(argc!=8)
  {
	  printf("This program must be run with file command-line arguments:");
	  printf("eg. ./Inclined_Exponential_Profile <n> <i> <R> <h> <t> <p> <output_name>");
      fflush(stdout);
	  return 1;
  }

  double sersic_n = strtod(argv[1], NULL);
  double inc_angle = strtod(argv[2], NULL);
  double scale_radius = strtod(argv[3], NULL);
  double scale_height = strtod(argv[4], NULL);
  double trunc_factor = strtod(argv[5], NULL);
  double pos_angle = strtod(argv[6], NULL);
  char * output_name = argv[7];

  printf("Sersic Index: %1.1f\n",sersic_n);
  printf("Inclination angle: %1.4f\n",inc_angle);
  printf("Scale radius: %3.2f\n",scale_radius);
  printf("Scale height: %3.2f\n",scale_height);
  printf("Truncation factor: %3.2f\n",trunc_factor);
  printf("Position angle: %1.4f\n",pos_angle);
  fflush(stdout);

  fftw_complex *pixelaverageft, *convmodelft, *PSFft, *dsmodelft, *dsmodel, *xshiftft, *yshiftft;
  double *pixelaverage, *image;
  double rfiducial, sum;
  float **rmodelft, *resampledmodelft, *thickdiskft;
  time_t t1, t2;
  int i, num, x, y, ip;
  int p;
  int hos;
  fftw_plan pixavplan, invplan;

  /* ******************************* */
  // step one:
  // make a circular galaxy surface brighness distribution and r2c FT it
  // this step is done once only at the start of the code, for each Sersic index
  // that we wish to simulate
  int mdim, hmdim, nsersic;
  double *model;
  fftw_complex *modelft;
  fftw_plan bigmodel;
  // set the dimension of the large input circular galaxy.  Note that this must be a large
  // value e.g. 16384 in order to avoid aliasing
  mdim = 16384;
  hmdim = 1 + mdim/2;
  model = (double*)calloc(mdim*mdim, sizeof(double));
  modelft = (fftw_complex*)fftw_malloc( mdim*hmdim*sizeof(fftw_complex) );
  bigmodel = fftw_plan_dft_r2c_2d(mdim, mdim, model, modelft, FFTW_ESTIMATE);
  // make a model with a nominal (fiducial) scalelength
  rfiducial = 30.;
  // make a set of models with different Sersic index values
  nsersic = 1;
  // store the real part of their fourier transform
  rmodelft = (float**)calloc(nsersic, sizeof(float*));
  for (i=0; i<nsersic; i++)
    {
      // allocate memory for this model FT
      rmodelft[i] = (float*)calloc(hmdim, sizeof(float));
      // make a large circular galaxy profile
      makecirculargalaxy(sersic_n, rfiducial, trunc_factor, mdim, model);
      // FFT
      fftw_execute(bigmodel);
      // convert FT complex to float Hankel and store with separate index for each model component
      makehankel(modelft, hmdim, rmodelft[i]);
    }
  // free memory not needed
  free(model);
  free(modelft);
  /* *********************** */

  // set the galaxy parameters - these would be set by the MCMC routine
  double overgalsize = scale_radius*oversampling; // major axis scalelength in pixels in oversampled arrays
  if (overgalsize >= rfiducial/sqrt(2.))
    {
      fflush(stdout);
      fprintf(stderr," error in specifying galaxy size, must be smaller than fiducial size/sqrt(2)\n");
      exit(0);
    }
  double overscaleheight = scale_height*oversampling;
  double xpos = 0.*oversampling;   // x position offset in oversampled pixels
  double ypos = 0.*oversampling;  // y position offset in oversampled pixels

  // allocate oversampled arrays
  // in future, make a selection of sizes to be chosen appropriately
  // to optimise the speed for small galaxy model generation

  int idim, odim, hdim;
  odim = 2*(int)(32.*scale_radius/2.0*pow(sersic_n,2)) ;
  idim = odim*oversampling;  // size of oversampled galaxy image
  hdim = 1 + idim/2;  // x-axis dimension of FFTW hermitian array

  // odim = idim;

  // allocate memory
  // pixel average function and its FT
  pixelaverage = (double*)calloc( idim*idim,sizeof(double) );
  pixelaverageft = (fftw_complex*)fftw_malloc( idim*hdim*sizeof(fftw_complex) );
  // x,y shift FTs
  yshiftft = (fftw_complex*)fftw_malloc( idim*sizeof(fftw_complex) );
  xshiftft = (fftw_complex*)fftw_malloc( hdim*sizeof(fftw_complex) );
  // r2c FFT of convolved model, stored as float
  resampledmodelft = (float*)calloc( idim*hdim,sizeof(float) );
  // r2c FFT of PSF, stored as complex
  PSFft = (fftw_complex*)calloc( idim*hdim,sizeof(fftw_complex) );
  // r2c FFT of thick disk convolving function, stored as complex
  thickdiskft = (float*)calloc( idim*hdim,sizeof(float) );
  // r2c FFT of oversampled, convolved model, stored as complex
  convmodelft = (fftw_complex*)fftw_malloc( idim*hdim*sizeof(fftw_complex) );
  // full FFT of downsampled model
  dsmodelft = (fftw_complex*)fftw_malloc( odim*odim*sizeof(fftw_complex) );
  // complex downsampled image domain model
  dsmodel = (fftw_complex*)calloc( odim*odim,sizeof(fftw_complex) );
  //dsmodel = (double*)calloc( odim*odim,sizeof(double) );
  // real part of downsampeld image domain model
  image = (double*)calloc( odim*odim,sizeof(double) );

  // complex downsampled image domain model
  //fftw_complex* rsmodel = (fftw_complex*)calloc( idim*idim,sizeof(fftw_complex) );
  //double* rsimage = (double*)calloc( idim*idim,sizeof(double) );

  // calculate fftw plans
  // pixelaverage plan
  pixavplan = fftw_plan_dft_r2c_2d(idim, idim, pixelaverage, pixelaverageft, FFTW_ESTIMATE);
  // inverse downsampled image plan
  invplan = fftw_plan_dft_2d(odim, odim, dsmodelft, dsmodel, FFTW_BACKWARD, FFTW_MEASURE);
  //invplan = fftw_plan_dft_c2r_2d(odim, odim, convmodelft, dsmodel, FFTW_ESTIMATE);

  //fftw_plan invplan2 = fftw_plan_dft_2d(odim, odim, convmodelft, rsmodel, FFTW_BACKWARD, FFTW_ESTIMATE);

  // fill up pixelaverage function
  // this function would also only be calculated once at the start of the galaxy measurement
  // set all values to zero
  for (ip=0; ip<idim*idim; ip++)
    {
      pixelaverage[ip]=0.;
    }
  // set a central box to a tophat function which sums to unity
  // set it to be centred in the array so we don't need to swap quadrants at the end
  hos = oversampling/2;
  int cen = idim/2;
  for (y=cen-hos; y<=cen+hos; y++)
    {
      for (x=cen-hos; x<=cen+hos; x++)
        {
          ip = y*idim + x;
          pixelaverage[ip] = 1./(double)(oversampling*oversampling);
        }
    }
  // create FT of pixelaverage
  fftw_execute(pixavplan);


  // create the FT of the PSF.  For now, just set the PSF to be a delta function in image domain
  // i.e. a uniform function in the Fourier domain
  // this function would be filled once for each galaxy component
  for (i=0; i<idim*hdim; i++) PSFft[i] = pixelaverageft[i];

  // choose which sersic index to make
  int sersicindex = 0; // this will make value of sersic index = 1

  double sini = sin(inc_angle);
  double sini_squared = sini*sini;

  double emod;

  if(sini_squared==0)
  {
	  emod = 0.;
  }
  else
  {
	  emod = (2. - sini_squared + 2*sqrt(1-sini_squared))/sini_squared;
  }

  double e1 = emod * cos(2*pos_angle);
  double e2 = emod * sin(2*pos_angle);

  // optional: run a timing test using a loop by setting a high value for num
  num = 1;
  t1 = time(NULL);
  int odimsq = odim*odim;
  for (i=0; i<num; i++)
    {

      /* ********************************************** */
      // this is the call to the C function that
      // generates a downsampled FT of galaxy model
      GenerateModel(e1, e2, overgalsize, overscaleheight, xpos, ypos, rfiducial, odim,
                    idim, mdim, rmodelft[sersicindex], resampledmodelft,
                    PSFft, thickdiskft, xshiftft, yshiftft, convmodelft, dsmodelft);
      /* ********************************************** */

      /* the following sections are all back in the Python function.
         in principle we could do the iFFT step inside the C function but this probably
         does not improve the speed and would mean also passing an fftw_plan into the C function */

      // make downsampled image inverse FFT
      fftw_execute(invplan);

      // take the real part (discard the imaginary part) and scale
      for (p=0; p<odimsq; p++)
        {
          image[p] = creal(dsmodel[p])/(odim*odim);
        }

      // end of timing loop
    }
  t2 = time(NULL);
  if (num>1)
    {
      printf(" time %g \n",difftime(t2,t1)/num);
    }


  sum = 0.;
  // take the real part (discard the imaginary part) and test the normalisation
  for (p=0; p<odimsq; p++)
    {
      sum += image[p];
    }
  printf(" sum %g \n",sum);

  // write out final output image to fits file
  remove(output_name);
  write2Dfits(output_name,image,odim,odim);

  exit(0);
}
