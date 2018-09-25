import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


lowthresh = Slider()

def update(val)
im = cv2.imread('key.jpg')
imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
thresh = cv2.Canny(imgray, 20, 100)
#ret, thresh = cv2.threshold(imgray, 80, 200,0)
image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
image = cv2.drawContours(image, contours, -1, (200,255,0), 1)