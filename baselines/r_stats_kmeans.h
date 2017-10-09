/*
 * Copyright (C) 2017  Lutz, Clemens <lutzcle@cml.li>
 * Author: Lutz, Clemens <lutzcle@cml.li>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef R_STATS_KMEANS_H
#define R_STATS_KMEANS_H

/*
 * x: data in column-major matrix format
 * pn: number of points [pn x pp]
 * pp: number of dimensions / features
 * cen: old centroids in column-major matrix format [pk x pp]
 * pk: number of clusters
 * cl: cluster labels vector [length pk]
 * pmaxiter: maximum number of iterations
 * nc: cluster mass vector [length pk]
 * wss: within-cluster sum-of-squares vector [length: pk]
 */
void kmeans_Lloyd(
    double *x,
    int *pn,
    int *pp,
    double *cen,
    int *pk,
    int *cl,
    int *pmaxiter,
    int *nc,
    double *wss
    );

void kmeans_MacQueen(
    double *x,
    int *pn,
    int *pp,
    double *cen,
    int *pk,
		int *cl,
    int *pmaxiter,
    int *nc,
    double *wss
    );

void kmeans_Lloyd_float(
    float *x,
    int *pn,
    int *pp,
    float *cen,
    int *pk,
    int *cl,
    int *pmaxiter,
    int *nc,
    float *wss
    );

#endif /* R_STATS_KMEANS_H */
