#!/usr/bin/env python2
import numpy as np
import cv2
from matplotlib import pyplot as plt
import glob
from string import ascii_lowercase as al
import math

def get_database_images(pathtoimage):
	database_images = []
	return database_images


def process_image(img):
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	th, im_th = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY_INV)
# Adaptive thresholding may be better for variable light conditions.
# im_th = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
	im_floodfill = im_th.copy()
	h, w = im_th.shape[:2]
	mask = np.zeros((h + 2, w + 2), np.uint8)
	cv2.floodFill(im_floodfill, mask, (0, 0), 255)
	im_floodfill_inv = cv2.bitwise_not(im_floodfill)
	im_out = im_th | im_floodfill_inv

	#out = [img, im_th, im_floodfill, im_floodfill_inv, im_out]
	#title = ['original', 'thresh', 'floodfill', 'notfloodfill', 'notfloodfillororiginal']
	#out = [im_out]
	#title = ['keyblank']
	return (im_out)


def remove_head_from_key(img):
	template = cv2.imread('template.jpg', cv2.COLOR_BGR2GRAY)
	w, h = template.shape[::-1]
	method = cv2.TM_SQDIFF
	# Apply template Matching
	res = cv2.matchTemplate(img, template, method)
	min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
	# If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
	if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
		top_left = min_loc
	else:
		top_left = max_loc
	bottom_right = (top_left[0] + w, top_left[1] + h)

	cv2.rectangle(img, top_left, bottom_right, 233, 2)
	return [template, img, res]


def get_new_key_image(pathtoimage):
	img = cv2.imread(pathtoimage)
	return img


def get_key_image_folder(pathtoimagefolder):
	images = [cv2.imread(file) for file in glob.glob(pathtoimagefolder.join("/*.jpg", sep=''))]
	return images


def display_image(img, titles=al):
	subplot = math.ceil(math.sqrt(len(img)))
	for i in xrange(len(img)):
		plt.subplot(subplot, subplot, i+1), plt.imshow(img[i], 'gray')
		plt.title(titles[i])
		plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
	plt.show()


def save_images(images, names=al, savepathfolder='', postfix='_key_test.jpg'):
	for img, name in zip(images, names):
		if len(savepathfolder) > 1:
			name = ''.join([savepathfolder, "/", name, postfix])
		else:
			name = ''.join([name, postfix])
		print(name)
		cv2.imwrite(name, img)


img = get_new_key_image('key3.jpg')
img = process_image(img)
images = remove_head_from_key(img)
#images, names = process_image(img)
display_image(images)
#save_images(images, names, "img_save")
