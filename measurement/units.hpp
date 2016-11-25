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
