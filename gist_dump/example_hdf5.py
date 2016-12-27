# -*- coding: utf 8 -*-
# Author: Kyle Kastner
# License: BSD 3-clause
from __future__ import division
import os
import numpy as np
import tables
import numbers
import fnmatch

def fetch_data():
    data_path = "/data/lisatmp3/Lessac_Blizzard2013_segmented/backup"
    partial_path = os.path.join("/Tmp/", os.getenv("USER"))
    hdf5_path = os.path.join(partial_path, "full_blizzard.h5")
    if not os.path.exists(hdf5_path):
        data_matches = []
        for root, dirnames, filenames in os.walk(data_path):
            for filename in fnmatch.filter(filenames, 'data_*.npy'):
                data_matches.append(os.path.join(root, filename))
        # sort in proper order
        data_matches = sorted(data_matches,
                              key=lambda x: int(
                                  x.split("/")[-1].split("_")[-1][0]))
        # setup tables
        sz = 32000
        compression_filter = tables.Filters(complevel=5, complib='blosc')
        hdf5_file = tables.openFile(hdf5_path, mode='w')
        data = hdf5_file.createEArray(hdf5_file.root, 'data',
                                      tables.Int16Atom(),
                                      shape=(0, sz),
                                      filters=compression_filter,)
        for na, f in enumerate(data_matches):
            print("Reading file %s" % (f))
            with open(f) as fp:
                # Array of arrays, ragged
                d = np.load(fp)
                for n, di in enumerate(d):
                    print("Processing line %i of %i" % (n, len(d)))
                    # Some of these are stereo??? wtf
                    if len(di.shape) < 2:
                        e = [r for r in range(0, len(di), sz)]
                        e.append(None)
                        starts = e[:-1]
                        stops = e[1:]
                        endpoints = zip(starts, stops)
                        for i, j in endpoints:
                            di_new = di[i:j]
                            # zero pad
                            if len(di_new) < sz:
                                di_large = np.zeros((sz,), dtype='int16')
                                di_large[:len(di_new)] = di_new
                                di_new = di_large
                            data.append(di_new[None])
        hdf5_file.close()
    hdf5_file = tables.openFile(hdf5_path, mode='r')
    data = hdf5_file.root.data
    X = data
    return X

if __name__ == "__main__":
    X = fetch_data()
    from IPython import embed; embed()
    raise ValueError()
