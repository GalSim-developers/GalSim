/* -*- c++ -*-
 * Copyright (c) 2012-2023 by the GalSim developers team on GitHub
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

//#define DEBUGLOGGING

#include <limits>
#include "PyBind11Helper.h"
#include "Std.h"

namespace galsim {

    static py::array MergeSorted(py::list& arrays)
    {
        dbg<<"Start MergeSorted: "<<arrays.size()<<std::endl;
        const int n_arrays = arrays.size();
        if (n_arrays == 0)
            throw std::runtime_error("No arrays provided to merge_sorted");
        assert(n_arrays > 0);
        py::array_t<double> a0 = arrays[0].cast<py::array_t<double> >();
        int n0 = a0.size();
        dbg<<"size of array 0 = "<<n0<<std::endl;

        // First figure out the maximum possible size of the return array.
        int max_ret_size = n0;
        for(int k=1; k<n_arrays; ++k) {
            py::array_t<double> ak = arrays[k].cast<py::array_t<double> >();
            int nk = ak.size();
            dbg<<"size of array "<<k<<" = "<<nk<<std::endl;
            // Check how far into the array, this one is identical to a0.
            // Do this from both sides.  Not least because in GalSim, the typical
            // way we use this includes a 2-element array which is often just the
            // first and last values of other arrays.
            const double* a0_p1 = static_cast<const double*>(a0.data());
            const double* ak_p1 = static_cast<const double*>(ak.data());
            const double* a0_p2 = a0_p1 + n0;
            const double* ak_p2 = ak_p1 + nk;
            while (a0_p1 != a0_p2 && ak_p1 != ak_p2 && *a0_p1 == *ak_p1) {
                ++a0_p1; ++ak_p1;
            }
            while (a0_p1 != a0_p2 && ak_p1 != ak_p2 && *(a0_p2-1) == *(ak_p2-1)) {
                --a0_p2; --ak_p2;
            }
            int n_left = ak_p2 - ak_p1;
            dbg<<"For array "<<k<<", "<<nk - n_left<<" elements are identical to a0\n";
            max_ret_size += n_left;
        }
        dbg<<"max_ret_size = "<<max_ret_size<<std::endl;
        if (max_ret_size == n0) {
            // Then arrays[0] already has all the values.  No need to merge.
            // (This is not terribly uncommon, and the early exit can save a lot of time!)
            dbg<<"Early exit.  a0 has everything.\n";
            return arrays[0].cast<py::array>();
        }

        // We actually merge these 1 at a time, since that's much simpler (and maybe even faster?).
        // At each step,
        // p0 is a pointer into the first array being merged (possibly a previous merge result)
        // p1 is a pointer into the second array being merged
        // p2 is a pointer into the resulting merged array.
        // Note: If more than 2 input arrays to merge, we might need a second temporary vector.
        //       This will be swapped with res as needed during the iteration.

        std::vector<double> res(max_ret_size);
        std::vector<double> res2(n_arrays == 2 ? 0 : max_ret_size);
        const double* p0 = static_cast<const double*>(a0.data());
        const double* p0_end = p0 + n0;
        double* p2 = res.data();
        int n_res = 0;
        assert(n_arrays > 1);  // Can't get here if len(arrays) == 1

        for(int k=1; k<n_arrays; ++k) {
            py::array_t<double> a1 = arrays[k].cast<py::array_t<double> >();
            const double* p1 = static_cast<const double*>(a1.data());
            const double* p1_end = p1 + a1.size();

            // Keep track of the previous value to be placed in the result array,
            // so we can raise an exception if an input array is not sorted.
            double prev = -std::numeric_limits<double>::max();

            while (p0 != p0_end && p1 != p1_end) {
                double x;
                // Select the smaller one.
                // Consider everything to be < nan
                if (*p1 < *p0 || std::isnan(*p0)) {
                    x = *p1++;
                    if (std::isnan(*p0) && std::isnan(x)) ++p0;
                } else {
                    x = *p0++;
                    // If p1 is also == x, increment that too.
                    // Note: !(*p1 != x) is so nan == nan here.
                    if (!(*p1 != x)) ++p1;
                }
                // skip duplicates (again, letting nan == nan here)
                if (!(x != prev)) continue;
                // Make sure the inputs make sense.
                if (x < prev) {
                    throw std::runtime_error("Arrays are not sorted");
                }
                *p2++ = prev = x;
            }

            // Now at least one of the two arrays are exhausted.  Fill the rest of res.
            while (p0 != p0_end) {
                double x = *p0++;
                if (!(x != prev)) continue;
                if (x < prev) {
                    throw std::runtime_error("Arrays are not sorted");
                }
                *p2++ = prev = x;
            }
            while (p1 != p1_end) {
                double x = *p1++;
                if (!(x != prev)) continue;
                if (x < prev) {
                    throw std::runtime_error("Arrays are not sorted");
                }
                *p2++ = prev = x;
            }
            assert(p2 <= res.data()+max_ret_size);
            // The final value of p2-res.data() is the relevant length of res.
            n_res = p2 - res.data();

            if (k+1 < n_arrays) {
                // Set up for the next pass through the loop.
                res.swap(res2);
                // Now res2 has the result of this loop.  Use that for a0 in next loop.
                p0 = res2.data();
                p0_end = p0 + n_res;
                p2 = res.data();
            }
        }
        dbg<<"Done. Final size = "<<n_res<<std::endl;
        // Finally, return res as a numpy array
        return py::array_t<double>(n_res, res.data());
    }

    void pyExportUtilities(py::module& _galsim)
    {
        _galsim.def("MergeSorted", &MergeSorted);
    }

} // namespace galsim
