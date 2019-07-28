import sys
from skimage.io import imread
from skimage.morphology import skeletonize
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
import PIL
import glob
from skimage import measure
from math import sqrt
import numpy as np
from skimage.measure import regionprops
import matplotlib.patches as patches
from skimage.transform import resize
import cv2
import math
from skimage import data,filters
import tensorflow as tf
import warnings
import heapq
#from rotate import rotate
#from train.conv import readModel
#from train.threeLayerConv import readModel
from train.threeLayerConvBn import readModel
#from train.test import readModel
#from train.dnn6and8 import readModel
#from train.dnn import readModel
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image
#from readCarSvcModel import rea
import imutils
from shapedetector import ShapeDetector