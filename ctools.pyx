#!/usr/bin/env python
# -*- coding: UTF-8 no BOM -*-

"""
   ________  ___  ___________    __
  / ____/\ \/ / |/ /_  __/   |  / /
 / /      \  /|   / / / / /| | / /
/ /___    / //   | / / / ___ |/ /___
\____/   /_//_/|_|/_/ /_/  |_/_____/

Copyright (c) 2015, C. Zhang.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1) Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.
2) Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation and/or
other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

DESCRIPTION
-----------
    kmeans
        Use k-means clustering to categorize data based on
        column features.
"""


import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sqrt
from scipy.spatial.distance import cdist
from numpy.random import normal, choice


##
# Functions to provide the kemans clustering capability
# reference:
#    https://gist.github.com/dwf/2200359
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.embedsignature(True)
cpdef tuple kmeans(DTYPE_t[:,:] data,
                   INTP_t k,
                   INTP_t max_iter=1000,
                   DTYPE_t threshold=1e-3):
    """
    DESCRIPTION
    -----------
    assign, centroid, iteration, converged, counts, inertia
        = kmeans(data, k, max_iter=1000)
        Use k-means on a data table where each column is a feature.
    PARAMETERS
    ----------
    data : cython.floating[:,:]
        observation vectors with column as feature.
    k : int
        target number of clusters/centroids.
    max_iter : int, optional
        maximum number of iterations allowed.
        (default set to 1000)
    RETURNS
    -------
    assign    : ndarray, 1-D, int
        cluster/centroid/mean ID for each feature vector.
    centroid  : ndarray, n-D, float64
        centroids/means for each cluster.
    iteration : int
        The number of iterations of k-means actually performed. This
        will be less than or equal to `max_iter` specified in the
        input arguments.
    converged : boolean/bint
        A boolean flag indicating whether or not the algorithm
        converged.
    counts    : ndarray, 1-D, int
        Number of vectors in each cluster.
    inertia   : np.float64
        a benchmark represent how well the clustering results are.
    """
    cdef INTP_t ndata = data.shape[0]   # total number of vectors
    cdef INTP_t nfeat = data.shape[1]   # number of feature used
    cdef INTP_t iteration = 0           # some control integers
    cdef bint converged = 0          # not converged as default

    cdef np.ndarray assign       = np.empty(ndata, dtype=intp)
    cdef np.ndarray assign_old   = np.empty(ndata, dtype=intp)
    cdef np.ndarray min_data     = np.empty(nfeat, dtype=DTYPE)
    cdef np.ndarray max_data     = np.empty(nfeat, dtype=DTYPE)
    cdef np.ndarray wgt_feat     = np.empty(nfeat, dtype=DTYPE)
    cdef np.ndarray centroid     = np.empty((k, nfeat), dtype=DTYPE)
    cdef np.ndarray counts       = np.zeros(k, dtype=intp)
    cdef np.ndarray seed_id      = np.empty(k, dtype=intp)
    cdef np.ndarray seed_va

    cdef INTP_t  i,j,n          # loop control
    cdef INTP_t  idx            # temp container for index
    cdef DTYPE_t seed_sum       # sum used in seeding
    cdef DTYPE_t diff, tmp      # difference before and after updating centroid
    cdef DTYPE_t inertia = 0.0  # inertia for final clustering results

    cdef INTP_t [:]   mv_seedid   = seed_id   # memoryview for smart seed
    cdef DTYPE_t[:,:] mv_data     = data      # memoryview for data
    cdef INTP_t [:]   mv_assign   = assign    # memoryview for assign
    cdef DTYPE_t[:,:] mv_centroid = centroid  # memoryview for centroid
    cdef INTP_t [:]   mv_counts   = counts    # memoryview for counts

    # generate initial centroid using smart seed function
    smart_seed(mv_data, mv_seedid)

    # assign the random selected feature vector as initial guess
    for i in range(k):
        idx = seed_id[i]
        for j in range(nfeat):
            centroid[i][j] = data[idx][j]

    # very first calculation
    compute_centroid(mv_data, mv_assign, mv_centroid, mv_counts)

    # keep updating the centroid until reach max_iter or converged
    while iteration < max_iter:
        iteration += 1
        for i in range(ndata):
            assign_old[i] = assign[i]  # record the old results
        # update centroid again
        compute_centroid(mv_data, mv_assign, mv_centroid, mv_counts)
        # test if the results changes
        diff = 0.0
        for i in range(ndata):
            tmp   = assign_old[i] - assign[i]
            diff += tmp * tmp
        if diff < threshold:
            converged = 1
            break

    # use first element for initial reference to get data range
    for j in range(nfeat):
        min_data[j] = data[0][j]
        max_data[j] = data[0][j]
    # quickly go through data to find min/max value for each feature
    for i in range(1, ndata):
        for j in range(nfeat):
            min_data[j] = DTYPE_min(min_data[j], data[i][j])
            max_data[j] = DTYPE_max(max_data[j], data[i][j])
    # calculate feature weight
    for j in range(nfeat):
        tmp = max_data[j] - min_data[j]
        wgt_feat[j] = 1.0/tmp if DTYPE_abs(tmp) > 1e-5 else 0.0

    # calculate inertia
    for i in range(ndata):
        idx = assign[i]  # who do I belongs to
        for j in range(nfeat):
            tmp = (data[i][j] - centroid[idx][j])*wgt_feat[j]
            inertia += tmp * tmp

    return assign, centroid, iteration, converged, counts, inertia


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void compute_centroid(DTYPE_t[:,:] data,
                                  INTP_t [:]   assign,
                                  DTYPE_t[:,:] centroid,
                                  INTP_t [:]   counts):
    """
    DESCRIPTION
    -----------
    compute_centroid(data, assign, centroid)
    Pure C function computing centroid, results are written back
    to assign, centroid.
    PARAMETERS
    ----------
    data: DTYPE_t
        Memoryviews of feature vectors, this is intended as a read only
        data set (although not enforced here).
    assign: INTP_t
        Memoryviews of the assignments, indexing each feature vector with
        cluster ID, namely belongs to which centroid. This array is updated
        by this function.
    centroid: DTYPE_t
        Memoryviews of the centroid with respect to given features in the
        feature vectors [data]. This array is updated by this function.
    NOTES
    -----
    This function uses the memoryview from Cython to update the data at
    each iteration.
    Still haven't considering the possible empty cluster case. Will come
    back to it later
    """
    cdef INTP_t ndata = data.shape[0]
    cdef INTP_t nfeat = data.shape[1]
    cdef INTP_t k     = centroid.shape[0]
    cdef np.ndarray m_dist = np.empty((ndata,k), dtype=DTYPE)
    cdef np.ndarray fv = np.empty(k, dtype=DTYPE)
    cdef np.ndarray centroid_tmp = np.zeros((k, nfeat), dtype=DTYPE)

    cdef INTP_t i, j
    cdef INTP_t idx     # temp container for centroid index
    cdef DTYPE_t dist  # temp container for distance

    # Before doing anything, zero counts
    for i in range(k):
        counts[i] = 0

    # First update the assignment
    m_dist = cdist(data, centroid)  # calculate distance matrix
    for i in range(ndata):
        fv = m_dist[i]  # find the assignment for vector i
        dist = fv[0]
        idx = 0
        # find the closet centroid
        for j in range(1, k):
            if fv[j] < dist:
                idx = j
        assign[i] = idx

    # Second calculate the new centroid
    for i in range(ndata):
        idx = assign[i]  # find out which centroid I belong to
        counts[idx] += 1  # add one count to that centroid
        for j in range(nfeat):
            centroid_tmp[idx][j] += data[i][j]  # add me in to that cluster

    # Update the centroid container with latest results
    for i in range(k):
        for j in range(nfeat):
            if counts[i] > 0:
                centroid[i][j] = centroid_tmp[i][j] / counts[i]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void smart_seed(DTYPE_t[:,:] data,
                            INTP_t [:]   seed_id):
    """
    DESCRIPTION
    -----------
    smart_seed(data, seed_id)
    use k-means++ method to do a smart seeding
        ref: https://en.wikipedia.org/wiki/K-means%2B%2B
    PARAMETERS
    ----------
    data: DTYPE_t
        Memoryviews of feature vectors, this is intended as a read only
        data set (although not enforced here).
    seed_id: INTP_t
        Memoryviews of the initial seed ID. This array is updated by
        this function.
    NOTES
    -----
    This initial seed generating algorithm does not ensure a perfect
    clustering result. To find the correct clustering, we still need
    to run kmeans clustering function several times.
    """
    cdef INTP_t   ndata = data.shape[0]
    cdef INTP_t   nfeat = data.shape[1]
    cdef INTP_t   k     = seed_id.shape[0]

    cdef np.ndarray weights  = np.empty(ndata, dtype=DTYPE)
    cdef np.ndarray ref_seed = np.zeros((1, nfeat), dtype=DTYPE)

    cdef INTP_t   i,j
    cdef INTP_t   id_ref     # hold the reference seed ID
    cdef DTYPE_t  norm


    # first, randomly select 1 seed
    seed_id[0] = choice(ndata, 1)

    for i in range(1, k):
        id_ref = seed_id[i-1]
        ref_seed[0] = data[id_ref]
        weights = cdist(data, ref_seed).flatten()
        norm = 0.0
        for j in range(ndata):
            weights[j] *= weights[j]
            norm       += weights[j]
        for j in range(ndata):
            weights[j] /= norm
        seed_id[i] = choice(ndata, 1, p=weights)

