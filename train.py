from pylab import *

caffe_root = '/nfs/bigbang/pratik18v/caffe-shihenw/'  # this file should be run from {caffe_root}/examples (otherwise change this line)

import sys
sys.path.insert(0, caffe_root + 'python')
import caffe

caffe.set_device(1)
caffe.set_mode_gpu()

solver = caffe.SGDSolver('prototxt/MPI/pose_solver.prototxt')

"""
for k, v in solver.net.blobs.items():
    print k, v.data.shape
    print "\n"
"""

#print solver.net.blobs['bbox'].data.shape
#print solver.net.blobs['data'].data.shape

solver.solve()
"""
solver.net.forward()
for i in range(solver.net.blobs['data'].data.shape[0]):
    im1 = solver.net.blobs['data'].data[i,0:3,:,:]
    im1 = np.transpose(im1,(1,2,0))
    plt.imsave('sample/image'+str(i)+'.jpg',im1)

    im2 = solver.net.blobs['bbox'].data[i,0:3,:,:]
    im2 = np.transpose(im2,(1,2,0))
    plt.imsave('sample/bb'+str(i)+'.jpg',im2)
"""
