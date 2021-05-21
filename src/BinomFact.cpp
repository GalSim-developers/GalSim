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

#include <vector>
#include "BinomFact.h"
#include "Std.h"

namespace galsim {

    double fact(int i)
    {
        assert(i>=0);
        static std::vector<double> f(10);
        static bool first=true;
        if (first) {
            f[0] = f[1] = 1.;
            for(int j=2;j<10;j++) f[j] = f[j-1]*(double)j;
            first = false;
        }
        if (i>=(int)f.size()) {
            for(int j=f.size();j<=i;j++)
                f.push_back(f[j-1]*(double)j);
            assert(i==(int)f.size()-1);
        }
        assert(i<(int)f.size());
        return f[i];
    }

    double sqrtfact(int i)
    {
        static std::vector<double> f(10);
        static bool first=true;
        if (first) {
            f[0] = f[1] = 1.;
            for(int j=2;j<10;j++) f[j] = f[j-1]*std::sqrt((double)j);
            first = false;
        }
        if (i>=(int)f.size())
            for(int j=f.size();j<=i;j++)
                f.push_back(f[j-1]*std::sqrt((double)j));
        assert(i<(int)f.size());
        return f[i];
    }

    double binom(int i,int j)
    {
        static std::vector<std::vector<double> > f(10);
        static bool first=true;
        if (first) {
            f[0] = std::vector<double>(1,1.);
            f[1] = std::vector<double>(2,1.);
            for(int i1=2;i1<10;i1++) {
                f[i1] = std::vector<double>(i1+1);
                f[i1][0] = f[i1][i1] = 1.;
                for(int j1=1;j1<i1;j1++) f[i1][j1] = f[i1-1][j1-1] + f[i1-1][j1];
            }
            first = false;
        }
        if (j<0 || j>i) return 0.;
        if (i>=(int)f.size()) {
            for(int i1=f.size();i1<=i;i1++) {
                f.push_back(std::vector<double>(i1+1,1.));
                for(int j1=1;j1<i1;j1++) f[i1][j1] = f[i1-1][j1-1] + f[i1-1][j1];
            }
            assert(i==(int)f.size()-1);
        }
        assert(i<(int)f.size());
        assert(j<(int)f[i].size());
        return f[i][j];
    }

    double sqrtn(int i)
    {
        static std::vector<double> f(10);
        static bool first=true;
        if (first) {
            for(int j=0;j<10;j++) f[j] = std::sqrt((double)j);
            first = false;
        }
        if (i>=(int)f.size())
            for(int j=f.size();j<=i;j++)
                f.push_back(std::sqrt((double)j));
        assert(i<(int)f.size());
        return f[i];
    }

}
