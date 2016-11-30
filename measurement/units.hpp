/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright (c) 2016, Lutz, Clemens <lutzcle@cml.li>
 */

#ifndef UNITS_HPP
#define UNITS_HPP

namespace Measurement {

struct Unit {
    enum u {
        Second,
        Millisecond,
        Microsecond,
        Nanosecond,
        Byte,
        Kilobyte,
        Megabyte,
        Gigabyte
    };
};
}

#endif /* UNITS_HPP */
