#!/usr/bin/env python

# pylearn2
from pylearn2.expr.preprocessing import global_contrast_normalize
import sys
import os
import numpy as np
import cPickle as pickle
import Image
from pylearn2.utils import serial
from theano import tensor as T
from theano import function
from pylearn2.datasets import preprocessing
from pylearn2.datasets import dense_design_matrix
import matplotlib.pyplot as pyplot

# ROS
import cv2
import rospy
import sensor_msgs.msg # import Image
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError

#import warnings
data_path='/home/william/pylearn2/data/'
load_path=data_path
save_path=data_path + 'cifar10/cifar-10-batches-py/prediction_batch'
class_list=[]

#THEANO_FLAGS="device=gpu,floatX=float32" python real_time_prediction.py my_dataset_old_small.pkl

print 'Initializing...'
model = serial.load(data_path + 'my_dataset_old_small.pkl')
X = model.get_input_space().make_theano_batch()
Y = model.fprop(X)
f = function([X], Y)
my_pca_preprocessor = pickle.load(open(data_path + '/cifar10/pylearn2_gcn_whitened/preprocessor.pkl','rb'))

'make cuda-convnet batches from images in the input dir; start numbering batches from 7'


class ImageClassify:
    NODE_NAME = "image_classification"

    def __init__(self):
        print 'Starting node...'
        self.discardCount = 0
        self.publisher = rospy.Publisher('/image_detect', String, queue_size=10)
        self.bridge = CvBridge()
        self.subscriber = rospy.Subscriber("/cv_camera/image_raw", sensor_msgs.msg.Image, self.callback)

    def callback(self, data):
        # discard some frames
        if self.discardCount < 5:
            self.discardCount += 1
            return

        self.discardCount = 0


        # do image conversion
        try:
            img = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError, e:
            print e

        # squeeze the image into 32x32 size
        img_resized = cv2.resize(img, (32, 32))
        cv2.imwrite(data_path + '/image_classification.png', img_resized)

        #cv2.imshow("ImageShow", img_resized)
        #cv2.waitKey(10)
        #cv2.destroyAllWindows()

        # do modification/packing & classification
        # packing
        makeBatch(load_path, save_path, class_list)

        # prediction
        msg = String()
        msg.data = real_time_prediction()

        self.publisher.publish(msg)


def makeBatch (load_path, save_path, class_list):
    print 'Packing...'
    data = []
    filenames = []
    file_list = os.listdir(load_path)

    for item in  file_list:
        if item.endswith(".png"):
            n = os.path.join(load_path, item)
            input = Image.open(n)
            arr = np.array(input, dtype='float32',order='C')
            im = np.fliplr(np.rot90(arr, k=3))
            data.append(im.T.flatten('C'))
            filenames.append(item)
         #   class_list.append(int(str(item)[0]))
    data = np.array(data)
    out_file = open(save_path, 'w+')
    flipDat = np.flipud(data)
    rotDat = np.rot90(flipDat, k=3)
    rotDat = np.transpose(rotDat)
    dic = {'batch_label':'batch of test', 'data':rotDat, 'labels':class_list, 'filenames':filenames}
    pickle.dump(dic, out_file)
    print filenames
    out_file.close()



def real_time_prediction():

    ### loading new images for classification starts here
    fo = open(save_path,'rb')   # batch path
    batch1 = pickle.load(fo)
    fo.close()

    xarr = np.array(batch1['data'],dtype='float32')
    xarr = global_contrast_normalize(xarr, scale=55.)

    no_of_row=len(batch1['data'])

    xdat = np.array(xarr.reshape((no_of_row,3,32,32)),dtype='float32')  #reshape first parameter = batch matrix no. of row
    xdat = np.transpose(xdat[:,:,:,:],(1,2,3,0))

    x = dense_design_matrix.DenseDesignMatrix(topo_view=xdat, axes = ['c', 0, 1, 'b'])
    x.apply_preprocessor(my_pca_preprocessor, can_fit = False)
    tarr = x.get_topological_view()
    #print tarr
    y = f(tarr)


    ###########searching max in matrix##################################################
    #j = no. of row in prediction_batch
    #i = no. of classes (0-9)
    #result=('airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck')
    result=('bottle','book','toy','pen','chair','coin','phone','hand','note','head')
    resultString=''

    for j in range(0,no_of_row):
      max_index=0
      max_no=y[j][0]
      #print max_no
      for i in range(0,10):
         if y[j][i]>max_no:
          max_no=y[j][i]
          max_index=i
        # print max_index
      print "======================"
      print 'Photo',j+1, ' max=', result[max_index]

      if j > 0:
          resultString += ','

      resultString += result[max_index]
    #print 'y =', y
    ###################################################################################3

    return resultString

def entryPoint():

    ic = ImageClassify()

    rospy.init_node('image_classification', anonymous=True)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()



if __name__ == '__main__':
    entryPoint()