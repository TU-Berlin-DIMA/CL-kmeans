/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 * 
 * 
 * Copyright (c) 2016, Lutz, Clemens <lutzcle@cml.li>
 */

#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <vector>
#include <memory>

#ifdef MATRIX_BOUNDSCHECK
#include <iostream>
#endif

namespace cle {

template <typename T, typename Talloc, typename INT, bool COL_MAJOR = true>
class Matrix {
public:
    using iterator = typename std::vector<T, Talloc>::iterator;

    Matrix()
        :
            x_dim_(0), y_dim_(0)
    {}

    Matrix(std::vector<T, Talloc>&& matrix, INT const x_dim, INT const y_dim)
        :
            raw_(std::move(matrix)), x_dim_(x_dim), y_dim_(y_dim)
    {}

    Matrix(Matrix<T, Talloc, INT, COL_MAJOR>& other)
        :
            raw_(other.raw_), x_dim_(other.x_dim_), y_dim_(other.y_dim_)
    {}

    Matrix(Matrix<T, Talloc, INT, COL_MAJOR>&& other)
        :
            raw_(std::move(other.raw_)),
            x_dim_(other.x_dim_), y_dim_(other.y_dim_)
    {}

    Matrix<T, Talloc, INT, COL_MAJOR>& operator= (Matrix<T, Talloc, INT, COL_MAJOR>&& other) {
        raw_ = std::move(other.raw_);
        x_dim_ = other.x_dim_;
        y_dim_ = other.y_dim_;

        return this;
    }

    Matrix<T, Talloc, INT, COL_MAJOR>& operator= (Matrix<T, Talloc, INT, COL_MAJOR>& other) {
        raw_ = other.raw_;
        x_dim_ = other.x_dim_;
        y_dim_ = other.y_dim_;

        return this;
    }

    inline iterator begin() {
        return raw_.begin();
    }

    inline iterator end() {
        return raw_.end();
    }

    void resize(INT const x_dim, INT const y_dim) {
        raw_.resize(x_dim * y_dim);
        x_dim_ = x_dim;
        y_dim_ = y_dim;
    }

    T* data() {
        return raw_.data();
    }

    T const* data() const {
        return raw_.data();
    }

    std::vector<T, Talloc>& get_data() {
        return raw_;
    }

    std::vector<T, Talloc> const& get_data() const {
        return raw_;
    }

    std::vector<T, Talloc>&& move_data() {
        return std::move(raw_);
    }

    inline T& operator() (INT const x, INT const y) {
#ifdef MATRIX_BOUNDSCHECK
        if (x >= x_dim_ || y >= y_dim_) {
            std::cerr << "Warning: Matrix out of bounds access: ("
                << x << "," << y << ") in ("
                << x_dim_ << "," << y_dim_ << ") matrix"
                << std::endl;
        }
#endif
        return raw_[y_dim_ * y + x];
    }

    inline T const& operator() (INT const x, INT const y) const {
#ifdef MATRIX_BOUNDSCHECK
        if (x >= x_dim_ || y >= y_dim_) {
            std::cerr << "Warning: Matrix out of bounds access: ("
                << x << "," << y << ") in ("
                << x_dim_ << "," << y_dim_ << ") matrix"
                << std::endl;
        }
#endif
        return raw_[y_dim_ * y + x];
    }

    inline INT size() const {
        return raw_.size();
    }

    inline INT rows() const {
        return x_dim_;
    }

    inline INT cols() const {
        return y_dim_;
    }

private:
    std::vector<T, Talloc> raw_;
    INT x_dim_;
    INT y_dim_;
};
}

#endif /* MATRIX_HPP */
