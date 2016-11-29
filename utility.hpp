/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright (c) 2016, Lutz, Clemens <lutzcle@cml.li>
 */

#ifndef UTILITY_HPP
#define UTILITY_HPP

#include <cstdint>

namespace Clustering {
class Utility {
public:

    /*
     * Mean Squared Error
     *
     * Input:
     *  begin:      Beginning of estimation container
     *  end:        End of estimation container
     *  reference:  Beginning of ground truth container
     *
     * Uses Kahan summation algorithm (quad-precision)
     */
    template <typename Iterator>
    static double mse(Iterator begin, Iterator end, Iterator reference)
    {
        double sum_squares = 0;
        double compensation = 0;
        uint64_t counter = 0;

        for (auto it = begin, ri = reference; it != end; ++it, ++ri)
        {
            double x = *it;
            double r = *ri;

            x = x - r;
            double y = x * x - compensation;
            double t = sum_squares + y;
            compensation = (t - sum_squares) - y;
            sum_squares = t;

            ++counter;
        }

        return sum_squares / counter;
    }
};
}

#endif /* UTILITY_HPP */
