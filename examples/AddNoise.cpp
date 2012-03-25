// Test of noise addition
#include <iostream>
#include "GalSim.h"

int
main(int argc, char *argv[])
{
    int dimension = std::atoi(argv[1]);
    std::string fitsname = argv[2];
    double value = std::atof(argv[3]);
    double gain = std::atof(argv[4]);
    double readNoise = std::atof(argv[5]);

    galsim::Image<float> img(dimension,dimension);
    img.fill(value);

    galsim::UniformDeviate ud;
    galsim::CcdNoise noise(ud, gain, readNoise);

    noise(img);

    double sum = 0.;
    double sumsq = 0.;
    int npix=0;
    for (int y=img.getYMin(); y<=img.getYMax(); y++)
        for (int x=img.getXMin(); x<=img.getXMax(); x++) {
            sum += img(x,y);
            sumsq += img(x,y)*img(x,y);
            npix++;
        }
    sum /= npix;
    sumsq /= npix;
    std::cout << "Mean: " << sum << std::endl;
    std::cout << "Variance " << sumsq - sum*sum << std::endl;
    std::cout << "Std deviation " << sqrt(sumsq-sum*sum) << std::endl;
}
