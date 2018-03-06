/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright (c) 2016-2018, Lutz, Clemens <lutzcle@cml.li>"
 */

CL_INT ccoord2ind(CL_INT dim, CL_INT row, CL_INT col) {
    return dim * col + row;
}
__kernel
void matrix_scalar(
        __global CL_TYPE_1 *const restrict matrix,
        CL_TYPE_2 const scalar,
        CL_INT const NUM_COLS,
        CL_INT const NUM_ROWS
        )
{
}

__kernel
void matrix_row_vector(
        __global CL_TYPE_1 *const restrict matrix,
        __global CL_TYPE_2 *const restrict vector,
        CL_INT const NUM_COLS,
        CL_INT const NUM_ROWS
        )
{
    CL_INT m_ind = ccoord2ind(
            NUM_ROWS,
            get_global_id(0),
            get_global_id(1));

    CL_TYPE_2 v = vector[get_global_id(0)];

    matrix[m_ind] = matrix[m_ind] BINARY_OP (CL_TYPE_1)v;
}

__kernel
void matrix_col_vector(
        __global CL_TYPE_1 *const restrict matrix,
        __global CL_TYPE_2 *const restrict vector,
        CL_INT const NUM_COLS,
        CL_INT const NUM_ROWS
        )
{
}

__kernel
void matrix_matrix(
        __global CL_TYPE_1 *const restrict fst_matrix,
        __global CL_TYPE_2 *const restrict snd_matrix,
        CL_INT const NUM_COLS,
        CL_INT const NUM_ROWS
        )
{
    CL_INT m_ind = get_global_id(0);

    CL_TYPE_2 snd = snd_matrix[m_ind];
    CL_TYPE_1 fst = fst_matrix[m_ind];
    CL_TYPE_1 res = fst BINARY_OP snd;
    fst_matrix[m_ind] = res;
}
