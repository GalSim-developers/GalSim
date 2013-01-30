// -*- c++ -*-
/*
 * Copyright 2012, 2013 The GalSim developers:
 * https://github.com/GalSim-developers
 *
 * This file is part of GalSim: The modular galaxy image simulation toolkit.
 *
 * GalSim is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * GalSim is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with GalSim.  If not, see <http://www.gnu.org/licenses/>
 */

#include "BinomFact.h"
#include "Std.h"
#include <vector>

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
