#ifndef DATA_GENERATOR_HPP
#define DATA_GENERATOR_HPP

#include <matrix.hpp>

#include <vector>
#include <cstdint>

#include <algorithm>
#include <random>

class DataGenerator {
public:
    using Matrix =
        cle::Matrix<uint32_t, std::allocator<uint32_t>, uint32_t>;

    Matrix const& ascending() {
        if (ascending_.size() == 0) {
            ascending_.resize(num_rows_, num_cols_);
            std::generate(
                    ascending_.begin(),
                    ascending_.end(),
                    [](){static uint32_t x = 0; return x++;}
                    );
        }

        return ascending_;
    }

    Matrix const& large() {
        if (large_.size() == 0) {
            large_.resize(num_rows_, num_cols_);
            std::fill(
                    large_.begin(),
                    large_.end(),
                    1 << 31
                    );
        }

        return large_;
    }

    Matrix const& random() {
        if (random_.size() == 0) {
            random_.resize(num_rows_, num_cols_);
            std::default_random_engine rgen;
            std::uniform_int_distribution<uint32_t> uniform;
            std::generate(
                    random_.begin(),
                    random_.end(),
                    [&](){return uniform(rgen);}
                    );
        }

        return random_;
    }

private:
    Matrix ascending_;
    Matrix large_;
    Matrix random_;

    uint64_t num_rows_ = 4;
    uint64_t num_cols_ = 256;
};

DataGenerator dgen;

#endif /* DATA_GENERATOR_HPP */
