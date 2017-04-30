from __future__ import division
from collections import namedtuple
from operator import itemgetter
from pprint import pformat
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os, time, pickle, random, itertools

DATA = './data/'
# window size
N = 2

# calculate means of windows of size n and take average, then normalize
# use means to calculate and save variances for later use
# returns normalized image and variances
def pre_proc(im):
    means = np.zeros(im.shape)
    variances = np.zeros(im.shape)
    for i in range(im.shape[0]):
        col_sums = None
        for j in range(im.shape[1]):
            # get mean of window around point
            if col_sums is None:
                # create col_sums array
                col_sums = np.zeros((2*N+1,2*N+1))
                for k in range(-N,N+1):
                    for l in range(-N,N+1):
                        if i+k>=0 and i+k<im.shape[0] and j+l>=0 and j+l<im.shape[1]:
                            col_sums[k+N,l+N] = im[i+k,j+l]
            else:
                for c in range(2*N+1):
                    # shift columns
                    if c+1 < 2*N+1:
                        col_sums[:,c] = col_sums[:,c+1]
                    else:
                        col_sums[:,c] = np.zeros(2*N+1)
                # now add new row
                if j+N<im.shape[1]:
                    for k in range(-N,N+1):
                        if i+k>=0 and i+k<im.shape[0]:
                            col_sums[k+N,2*N] = im[i+k,j+N]
            count = np.count_nonzero(col_sums)
            means[i,j] = np.sum(col_sums)/count

            # calculate variances
            def f(x):
                return x**2 - means[i,j]**2
            f = np.vectorize(f)

            variances[i,j] = np.sum(f(col_sums))/count

    return (im-means,variances)

def main():
    # load and normalize ims
    im0 = np.array(Image.open(DATA+'map/im0.pgm'))
    im1 = np.array(Image.open(DATA+'map/im1.pgm'))

    print im0[200,200]

    (im0, im0vars) = pre_proc(im0)
    (im1, im1vars) = pre_proc(im1)

    print im0[200,200]
    print 'done'

if __name__ == '__main__':
    main()
