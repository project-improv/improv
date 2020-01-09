from builtins import str
from builtins import range
from past.utils import old_div

import base64
import cv2 as cv
from IPython.display import HTML
from math import sqrt, ceil
import matplotlib as mpl
import matplotlib.cm as cm
from matplotlib.widgets import Slider
import matplotlib.pyplot as plt
import numpy as np
import pylab as pl
from scipy.ndimage.measurements import center_of_mass
from scipy.ndimage.filters import median_filter
from scipy.sparse import issparse, spdiags, coo_matrix, csc_matrix
from skimage.measure import find_contours
import sys
from tempfile import NamedTemporaryFile
from typing import Dict
from warnings import warn
import holoviews as hv
import functools as fct
import time
from os.path import expanduser
import os

try:
    cv2.setNumThreads(0)
except:
    pass

try:
    import bokeh
    import bokeh.plotting as bpl
    from bokeh.models import CustomJS, ColumnDataSource, Range1d
except:
    print("Bokeh could not be loaded. Either it is not installed or you are not running within a notebook")


def get_contours(A, dims, thr=0.9, thr_method='nrg', swap_dim=False):
    """Gets contour of spatial components and returns their coordinates

     Args:
         A:   np.ndarray or sparse matrix
                   Matrix of Spatial components (d x K)

             dims: tuple of ints
                   Spatial dimensions of movie (x, y[, z])

             thr: scalar between 0 and 1
                   Energy threshold for computing contours (default 0.9)

             thr_method: [optional] string
                  Method of thresholding:
                      'max' sets to zero pixels that have value less than a fraction of the max value
                      'nrg' keeps the pixels that contribute up to a specified fraction of the energy

     Returns:
         Coor: list of coordinates with center of mass and
                contour plot coordinates (per layer) for each component
    """

    if 'csc_matrix' not in str(type(A)):
        A = csc_matrix(A)
    d, nr = np.shape(A)
    # if we are on a 3D video
    '''
    if len(dims) == 3:
        d1, d2, d3 = dims
        x, y = np.mgrid[0:d2:1, 0:d3:1]
    else:
    ''' # remove becuase we want frame by frame
    d1, d2 = dims
    x, y = np.mgrid[0:d1:1, 0:d2:1]

    coordinates = []

    # for each patches
    patch_data_time=0
    indx_time= 0
    thr_method_time= 0
    swap_dim_time= 0
    vtx_loop_time=0
    pars_add_time=0
    vertices_time= 0
    atleast_time=0
    append_time= 0
    close_coords_time= 0
    newpt_time= 0
    vtx_time= 0
    concat_time=0
    atleastconcat_time=0
    for i in range(nr):
        # we compute the cumulative sum of the energy of the Ath component that has been ordered from least to highest
        t0= time.time()
        patch_data = A.data[A.indptr[i]:A.indptr[i + 1]]
        t1=  time.time()
        patch_data_time+= t1-t0

        t2= time.time()
        indx = np.argsort(patch_data)[::-1]
        t3= time.time()
        indx_time += t3-t2
        if thr_method == 'nrg':
            cumEn = np.cumsum(patch_data[indx]**2)
            # we work with normalized values
            cumEn /= cumEn[-1]
            Bvec = np.ones(d)
            # we put it in a similar matrix
            Bvec[A.indices[A.indptr[i]:A.indptr[i + 1]][indx]] = cumEn
        else:
            if thr_method != 'max':
                warn("Unknown threshold method. Choosing max")
            Bvec = np.zeros(d)
            Bvec[A.indices[A.indptr[i]:A.indptr[i + 1]]] = patch_data / patch_data.max()
        
        t4=time.time()

        thr_method_time = t4-t3
        if swap_dim:
            Bmat = np.reshape(Bvec, dims, order='C')
        else:
            Bmat = np.reshape(Bvec, dims, order='F')
        #import pdb; pdb.set_trace()

        #if (i==0):
            #visualize patch_data, contours
        #    patch_data=np.append(patch_data, 0)
        #    patch_data= np.append([0], patch_data)
        #    Amat= np.reshape(patch_data, dims, order='C')
        #    plt.subplot(1, 2, 1)
        #    plt.imshow(Amat)
        #    plt.subplot(1, 2, 2)
        #    plt.imshow(Bmat)
        #    plt.show()


        t5=time.time()
        swap_dim_time+= t5-t4
        # for each dimensions we draw the contour
        t6=time.time()
        retval, thresh= cv.threshold(Bmat.T, thr, 1, cv.THRESH_BINARY)
        thresh= thresh.astype(np.uint8)
        vertices, hierarchy= cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        #verticesPrev = find_contours(Bmat.T, thr)

        #if (i==0):
        #    image= Bmat.T
        #    for i,c in enumerate(verticesPrev):
        #        #c = np.array(c)
        #        c_img= cv.fillConvexPoly(image, c, (255,255,255,25))
        #    plt.imshow(image)
        #    plt.title('SkiImage contours')
        #    plt.show()

        #if (i==0):
        #    c_img = cv.drawContours(Bmat.T, vertices, -1, (0, 255, 0), 2)
        #    plt.figure()
        #    plt.imshow(c_img)
        #    plt.title('OpenCV contours')
        #    plt.show()
        #    fig, ax = plt.subplots()
        #    ax.imshow(Bmat.T)
        #    for n, contour in enumerate(verticesPrev):
        #        ax.plot(contour[:, 1], contour[:, 0], color='black', linewidth=2)
        #    ax.axis('image')
        #    ax.set_xticks([])
        #    ax.set_yticks([])
        #    plt.title('SkiImage contours')
        #    plt.show()
            
        t7= time.time()
        vertices_time += t7-t6
        # this fix is necessary for having disjoint figures and borders plotted correctly
        v = np.atleast_2d([np.nan, np.nan])
        t8= time.time()
        atleast_time+= t8-t7
        for _, vtx in enumerate(vertices):
            t11= time.time()

            #num_close_coords = np.isclose(vtx[0, :], vtx[-1, :])
            num_close_coords= [False, False]
            vtx= np.reshape(vtx, (vtx.shape[0], vtx.shape[2]))
            for i in range(2):
                num_close_coords[i]= (np.absolute(vtx[0, i]- vtx[-1, i])<=(0.00001+0.00000001*np.absolute(vtx[-1, i])))

            # implemented np.isclose manually, saved about 100 msec
            t12= time.time()
            close_coords_time+=t12-t11
            num_close_coords=np.sum(num_close_coords)
            if num_close_coords < 2:
                if num_close_coords == 0:
                    # case angle
                    t13=time.time()
                    newpt = np.round(old_div(vtx[-1, :], [d2, d1])) * [d2, d1]
                    t14= time.time()
                    newpt_time+= t14-t13
                    vtx = np.vstack((vtx, newpt[np.newaxis, :]))
                    t15= time.time()

                    vtx_time += t15-t14
                else:
                    # case one is border
                    t16=  time.time()
                    vtx = np.vstack((vtx, vtx[0, np.newaxis]))
                    t17= time.time()
                    vtx_time += t17-t16

            t18= time.time()
            v = np.vstack((v, vtx, np.atleast_2d([np.nan, np.nan]))) 

            t19=time.time()
            #v = np.vstack((v,  )))
            t20= time.time()
            concat_time+= t19-t18
            atleastconcat_time+= t20-t19
        t9= time.time()
        vtx_loop_time += t9-t8
        v[:, [0, 1]]= v[:, [1, 0]]
        coordinates.append(v)
        t10=  time.time()
        append_time+= t10-t9 
    print("nr: "+str(nr))
    print("patch_data time: " + str(patch_data_time))
    print("indx time: " + str(indx_time))
    print("Thr_method_time: "+ str(thr_method_time))
    print("swap_dim_time: "+ str(swap_dim_time))
    print("Vertices time: "+str(vertices_time))
    print("Atleast_time: "+ str(atleast_time))
    print("vtx loop time: "+ str(vtx_loop_time))
    print("append time: "+ str(append_time))
    print("close_coords_time: "+ str(close_coords_time))
    print("newpt_time: "+ str(newpt_time))
    print("vtx time: " + str(vtx_time))
    print("concat time: "+ str(concat_time))
    print("atleast concat time: "+ str(atleastconcat_time))
    return coordinates

if __name__ == '__main__':
    cwd = os.getcwd()
    A= np.loadtxt(cwd+'/data/A')
    tb= time.time()
    for i in range(30):
        X= get_contours(A, (440, 256))
    #X= get_contours(A, (440, 256))
    print(X[5])
    print(len(X[5]))
    ta=time.time()
    print("Single execute average: "+ str((ta-tb)/30))