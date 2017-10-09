/*
 *  R : A Computer Language for Statistical Data Analysis
 *  Copyright (C) 2004   The R Core Team.
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, a copy is available at
 *  https://www.R-project.org/Licenses/
 */

/*
 * Source: https://github.com/wch/r-source/blob/trunk/src/library/stats/src/kmeans.c
 * Version: 1f16a9b
 *
 * Parameters:
 *
 * x:        data in column-major matrix format
 * pn:       number of points [pn x pp]
 * pp:       number of dimensions / features
 * cen:      centroids in column-major matrix format [pk x pp]
 * pk:       number of clusters
 * cl:       cluster labels vector [length pk]
 * pmaxiter: maximum number of iterations
 * nc:       cluster mass vector [length pk]
 * wss:      within-cluster sum-of-squares vector [length: pk]
 */

#include "r_stats_kmeans.h"

#include <stdbool.h>
#include <float.h>

void kmeans_Lloyd(double *x, int *pn, int *pp, double *cen, int *pk, int *cl,
		  int *pmaxiter, int *nc, double *wss)
{
    int n = *pn, k = *pk, p = *pp, maxiter = *pmaxiter;
    int iter, i, j, c, it, inew = 0;
    double best, dd, tmp;
    bool updated;

    for(i = 0; i < n; i++) cl[i] = -1;
    for(iter = 0; iter < maxiter; iter++) {
	updated = false;
	for(i = 0; i < n; i++) {
	    /* find nearest centre for each point */
	    best = DBL_MAX;
	    for(j = 0; j < k; j++) {
		dd = 0.0;
		for(c = 0; c < p; c++) {
		    tmp = x[i+n*c] - cen[j+k*c];
		    dd += tmp * tmp;
		}
		if(dd < best) {
		    best = dd;
		    inew = j+1;
		}
	    }
	    if(cl[i] != inew) {
		updated = true;
		cl[i] = inew;
	    }
	}
	if(!updated) break;
	/* update each centre */
	for(j = 0; j < k*p; j++) cen[j] = 0.0;
	for(j = 0; j < k; j++) nc[j] = 0;
	for(i = 0; i < n; i++) {
	    it = cl[i] - 1; nc[it]++;
	    for(c = 0; c < p; c++) cen[it+c*k] += x[i+c*n];
	}
	for(j = 0; j < k*p; j++) cen[j] /= nc[j % k];
    }

    *pmaxiter = iter + 1;
    for(j = 0; j < k; j++) wss[j] = 0.0;
    for(i = 0; i < n; i++) {
	it = cl[i] - 1;
	for(c = 0; c < p; c++) {
	    tmp = x[i+n*c] - cen[it+k*c];
	    wss[it] += tmp * tmp;
	}
    }
}

void kmeans_MacQueen(double *x, int *pn, int *pp, double *cen, int *pk,
		     int *cl, int *pmaxiter, int *nc, double *wss)
{
    int n = *pn, k = *pk, p = *pp, maxiter = *pmaxiter;
    int iter, i, j, c, it, inew = 0, iold;
    double best, dd, tmp;
    bool updated;

    /* first assign each point to the nearest cluster centre */
    for(i = 0; i < n; i++) {
	best = DBL_MAX;
	for(j = 0; j < k; j++) {
	    dd = 0.0;
	    for(c = 0; c < p; c++) {
		tmp = x[i+n*c] - cen[j+k*c];
		dd += tmp * tmp;
	    }
	    if(dd < best) {
		best = dd;
		inew = j+1;
	    }
	}
	if(cl[i] != inew) cl[i] = inew;
    }
   /* and recompute centres as centroids */
    for(j = 0; j < k*p; j++) cen[j] = 0.0;
    for(j = 0; j < k; j++) nc[j] = 0;
    for(i = 0; i < n; i++) {
	it = cl[i] - 1; nc[it]++;
	for(c = 0; c < p; c++) cen[it+c*k] += x[i+c*n];
    }
    for(j = 0; j < k*p; j++) cen[j] /= nc[j % k];

    for(iter = 0; iter < maxiter; iter++) {
	updated = false;
	for(i = 0; i < n; i++) {
	    best = DBL_MAX;
	    for(j = 0; j < k; j++) {
		dd = 0.0;
		for(c = 0; c < p; c++) {
		    tmp = x[i+n*c] - cen[j+k*c];
		    dd += tmp * tmp;
		}
		if(dd < best) {
		    best = dd;
		    inew = j;
		}
	    }
	    if((iold = cl[i] - 1) != inew) {
		updated = true;
		cl[i] = inew + 1;
		nc[iold]--; nc[inew]++;
		/* update old and new cluster centres */
		for(c = 0; c < p; c++) {
		    cen[iold+k*c] += (cen[iold+k*c] - x[i+n*c])/nc[iold];
		    cen[inew+k*c] += (x[i+n*c] - cen[inew+k*c])/nc[inew];
		}
	    }
	}
	if(!updated) break;
    }

    *pmaxiter = iter + 1;
    for(j = 0; j < k; j++) wss[j] = 0.0;
    for(i = 0; i < n; i++) {
	it = cl[i] - 1;
	for(c = 0; c < p; c++) {
	    tmp = x[i+n*c] - cen[it+k*c];
	    wss[it] += tmp * tmp;
	}
    }
}

void kmeans_Lloyd_float(float *x, int *pn, int *pp, float *cen, int *pk, int *cl,
		  int *pmaxiter, int *nc, float *wss)
{
    int n = *pn, k = *pk, p = *pp, maxiter = *pmaxiter;
    int iter, i, j, c, it, inew = 0;
    float best, dd, tmp;
    bool updated;

    for(i = 0; i < n; i++) cl[i] = -1;
    for(iter = 0; iter < maxiter; iter++) {
	updated = false;
	for(i = 0; i < n; i++) {
	    /* find nearest centre for each point */
	    best = FLT_MAX;
	    for(j = 0; j < k; j++) {
		dd = 0.0;
		for(c = 0; c < p; c++) {
		    tmp = x[i+n*c] - cen[j+k*c];
		    dd += tmp * tmp;
		}
		if(dd < best) {
		    best = dd;
		    inew = j+1;
		}
	    }
	    if(cl[i] != inew) {
		updated = true;
		cl[i] = inew;
	    }
	}
	if(!updated) break;
	/* update each centre */
	for(j = 0; j < k*p; j++) cen[j] = 0.0;
	for(j = 0; j < k; j++) nc[j] = 0;
	for(i = 0; i < n; i++) {
	    it = cl[i] - 1; nc[it]++;
	    for(c = 0; c < p; c++) cen[it+c*k] += x[i+c*n];
	}
	for(j = 0; j < k*p; j++) cen[j] /= nc[j % k];
    }

    *pmaxiter = iter + 1;
    for(j = 0; j < k; j++) wss[j] = 0.0;
    for(i = 0; i < n; i++) {
	it = cl[i] - 1;
	for(c = 0; c < p; c++) {
	    tmp = x[i+n*c] - cen[it+k*c];
	    wss[it] += tmp * tmp;
	}
    }
}
