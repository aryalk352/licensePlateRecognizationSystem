import importlib, importlib.util
import os
import cv2
import re
import numpy as np
import zipfile
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import shuffle
import shutil
from keras import models
import random
import glob
import urllib.request
import tarfile
import six.moves.urllib as urllib
import sys
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
sys.path.append("/mnt/c/Users/kitna/Desktop/college_routine/fourth_semester/Deep_Learning/assignment3/models/research/object_detection")
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


def module_from_file(module_name, file_path):
	spec = importlib.util.spec_from_file_location(module_name, file_path)
	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)
	return module

viewPath = "/mnt/c/Users/kitna/Desktop/college_routine/fourth_semester/Deep_Learning/assignment3/main/"

PATH_TO_CKPT_model1 = "/mnt/c/Users/kitna/Desktop/college_routine/fourth_semester/Deep_Learning/assignment3/fine_tuned_model/frozen_inference_graph.pb"
PATH_TO_CKPT_model2 = "/mnt/c/Users/kitna/Desktop/college_routine/fourth_semester/Deep_Learning/assignment3/fine_tuned_model_2/frozen_inference_graph.pb"
PATH_TO_LABELS_license_plate = "/mnt/c/Users/kitna/Desktop/college_routine/fourth_semester/Deep_Learning/assignment3/dataset/train/label_map.pbtxt"
PATH_TO_LABELS_character_detection = "/mnt/c/Users/kitna/Desktop/college_routine/fourth_semester/Deep_Learning/assignment3/second_model1/label_map.pbtxt"
path_to_cropped_images = '/mnt/c/Users/kitna/Desktop/college_routine/fourth_semester/Deep_Learning/assignment3/dataset/cropped_images'
	
class MainProject:
	
	def __init__(self):
		print("class created");
		

	def test(self):
		view1 = module_from_file("sample.py",viewPath+"sample.py")
		view1.displayView()
		return "test test";
		
	def crop_images(self,img_path,output_dict):
		inx = list(output_dict['detection_scores']).index(max(output_dict['detection_scores']))
		bounding_box = output_dict['detection_boxes'][inx]
		image_save_to = path_to_cropped_images +"/"+ img_path.split("/")[-1]
		image_sel = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
		height,width = image_sel.shape
		y1=int(bounding_box[0]*height)
		x1=int(bounding_box[1]*width)
		y2=int(bounding_box[2]*height)
		x2=int(bounding_box[3]*width)
		new_img = cv2.resize(image_sel[y1:y2,x1:x2],(200,90))
		(thresh, im_bw) = cv2.threshold(new_img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
		cv2.imwrite(image_save_to,im_bw)
		return image_save_to;
		
	def get_num_classes(self,pbtxt_fname):
		from object_detection.utils import label_map_util
		label_map = label_map_util.load_labelmap(pbtxt_fname)
		categories = label_map_util.convert_label_map_to_categories(
			label_map, max_num_classes=90, use_display_name=True)
		category_index = label_map_util.create_category_index(categories)
		return len(category_index.keys())
		
	def getValues(self,output_dict,category_index):
		#print(category_index)
		#output_dict['detection_boxes'],output_dict['detection_classes'],output_dict['detection_scores'],
		j2 = [list(output_dict['detection_scores']).index(i) for i in output_dict['detection_scores'] if i >= .45]
		a= min(j2)
		b=max(j2)
		plateNum = ""
		print("-------------------------------------------------------------------------")
		for cat in output_dict['detection_classes'][a:b]:
			plateNum=plateNum+category_index[cat]["name"]
		print(plateNum)
		print("scores",output_dict['detection_scores'][a:b])
		x1 = np.round(output_dict['detection_boxes'][a:b][0:b,1:2],2).reshape(-1)
		x1_sort = np.sort(x1)
		print(x1)
		print(x1_sort)
		dupes = [x for n, x in enumerate(x1) if x in x1[:n]]
		multi = {}
		for d in dupes:
			sc = []
			for inx,val in enumerate(x1):
				if d==val:
					sc.append(output_dict['detection_scores'][inx])
			multi[d]=list(output_dict['detection_scores']).index(max(sc))
		print(multi)
		a1=[]
		for vale in np.unique(x1_sort):
			if dupes.count(vale)>0:
				a1.append(multi[vale])
			else:
				a1.append(list(x1).index(vale))
		#a2 = [list(x1).index(i) for i in np.unique(x1_sort)]
		print(a1)
		corOrd = ""
		for inx in a1:
			corOrd=corOrd+category_index[output_dict['detection_classes'][inx]]['name']
		print(corOrd)
		return corOrd
	
	def get_license_plate(self,image_path):
		print(image_path)
		detection_graph = tf.Graph()
		with detection_graph.as_default():
			od_graph_def = tf.GraphDef()
			with tf.gfile.GFile(PATH_TO_CKPT_model1, 'rb') as fid:
				serialized_graph = fid.read()
				od_graph_def.ParseFromString(serialized_graph)
				tf.import_graph_def(od_graph_def, name='')


		label_map = label_map_util.load_labelmap(PATH_TO_LABELS_license_plate)
		num_classes = self.get_num_classes(PATH_TO_LABELS_license_plate)
		categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_classes, use_display_name=True)
		category_index = label_map_util.create_category_index(categories)


		def load_image_into_numpy_array(image):
			(im_width, im_height) = image.size
			return np.array(image.getdata()).reshape(
				(im_height, im_width, 3)).astype(np.uint8)

		# Size, in inches, of the output images.
		#IMAGE_SIZE = (12, 8)

		def run_inference_for_single_image(image, graph):
			with graph.as_default():
				with tf.Session() as sess:
					# Get handles to input and output tensors
					ops = tf.get_default_graph().get_operations()
					all_tensor_names = {
						output.name for op in ops for output in op.outputs}
					tensor_dict = {}
					for key in [
						'num_detections', 'detection_boxes', 'detection_scores',
						'detection_classes', 'detection_masks'
					]:
						tensor_name = key + ':0'
						if tensor_name in all_tensor_names:
							tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
								tensor_name)
					if 'detection_masks' in tensor_dict:
						# The following processing is only for single image
						detection_boxes = tf.squeeze(
							tensor_dict['detection_boxes'], [0])
						detection_masks = tf.squeeze(
							tensor_dict['detection_masks'], [0])
						# Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
						real_num_detection = tf.cast(
							tensor_dict['num_detections'][0], tf.int32)
						detection_boxes = tf.slice(detection_boxes, [0, 0], [
												   real_num_detection, -1])
						detection_masks = tf.slice(detection_masks, [0, 0, 0], [
												   real_num_detection, -1, -1])
						detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
							detection_masks, detection_boxes, image.shape[0], image.shape[1])
						detection_masks_reframed = tf.cast(
							tf.greater(detection_masks_reframed, 0.5), tf.uint8)
						# Follow the convention by adding back the batch dimension
						tensor_dict['detection_masks'] = tf.expand_dims(
							detection_masks_reframed, 0)
					image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

					# Run inference
					output_dict = sess.run(tensor_dict,
										   feed_dict={image_tensor: np.expand_dims(image, 0)})

					# all outputs are float32 numpy arrays, so convert types as appropriate
					output_dict['num_detections'] = int(
						output_dict['num_detections'][0])
					output_dict['detection_classes'] = output_dict[
						'detection_classes'][0].astype(np.uint8)
					output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
					output_dict['detection_scores'] = output_dict['detection_scores'][0]
					if 'detection_masks' in output_dict:
						output_dict['detection_masks'] = output_dict['detection_masks'][0]
			return output_dict

		image = Image.open(image_path)
		# the array based representation of the image will be used later in order to prepare the
		# result image with boxes and labels on it.
		image_np = load_image_into_numpy_array(image)
		# Expand dimensions since the model expects images to have shape: [1, None, None, 3]
		image_np_expanded = np.expand_dims(image_np, axis=0)
		# Actual detection.
		output_dict = run_inference_for_single_image(image_np, detection_graph)
		cropped_image_path = self.crop_images(image_path,output_dict)
		
		return cropped_image_path;
		
	def get_number_value(self,cropped_image_path):
		print("start of processing cropped image")
		detection_graph = tf.Graph()
		with detection_graph.as_default():
			od_graph_def = tf.GraphDef()
			with tf.gfile.GFile(PATH_TO_CKPT_model2, 'rb') as fid:
				serialized_graph = fid.read()
				od_graph_def.ParseFromString(serialized_graph)
				tf.import_graph_def(od_graph_def, name='')


		print("start to process label map")
		label_map = label_map_util.load_labelmap(PATH_TO_LABELS_character_detection)
		num_classes = self.get_num_classes(PATH_TO_LABELS_character_detection)
		categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_classes, use_display_name=True)
		category_index = label_map_util.create_category_index(categories)


		def load_image_into_numpy_array(image):
			(im_width, im_height) = image.size
			return np.array(image.getdata()).reshape(
				(im_height, im_width, 3)).astype(np.uint8)

		# Size, in inches, of the output images.
		IMAGE_SIZE = (8, 6)


		def run_inference_for_single_image(image, graph):
			with graph.as_default():
				with tf.Session() as sess:
					# Get handles to input and output tensors
					ops = tf.get_default_graph().get_operations()
					all_tensor_names = {
						output.name for op in ops for output in op.outputs}
					tensor_dict = {}
					for key in [
						'num_detections', 'detection_boxes', 'detection_scores',
						'detection_classes', 'detection_masks'
					]:
						tensor_name = key + ':0'
						if tensor_name in all_tensor_names:
							tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
								tensor_name)
					if 'detection_masks' in tensor_dict:
						# The following processing is only for single image
						detection_boxes = tf.squeeze(
							tensor_dict['detection_boxes'], [0])
						detection_masks = tf.squeeze(
							tensor_dict['detection_masks'], [0])
						# Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
						real_num_detection = tf.cast(
							tensor_dict['num_detections'][0], tf.int32)
						detection_boxes = tf.slice(detection_boxes, [0, 0], [
												   real_num_detection, -1])
						detection_masks = tf.slice(detection_masks, [0, 0, 0], [
												   real_num_detection, -1, -1])
						detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
							detection_masks, detection_boxes, image.shape[0], image.shape[1])
						detection_masks_reframed = tf.cast(
							tf.greater(detection_masks_reframed, 0.5), tf.uint8)
						# Follow the convention by adding back the batch dimension
						tensor_dict['detection_masks'] = tf.expand_dims(
							detection_masks_reframed, 0)
					image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

					# Run inference
					output_dict = sess.run(tensor_dict,
										   feed_dict={image_tensor: np.expand_dims(image, 0)})

					# all outputs are float32 numpy arrays, so convert types as appropriate
					output_dict['num_detections'] = int(
						output_dict['num_detections'][0])
					output_dict['detection_classes'] = output_dict[
						'detection_classes'][0].astype(np.uint8)
					output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
					output_dict['detection_scores'] = output_dict['detection_scores'][0]
					if 'detection_masks' in output_dict:
						output_dict['detection_masks'] = output_dict['detection_masks'][0]
			return output_dict


		image = Image.open(cropped_image_path)
		# the array based representation of the image will be used later in order to prepare the
		# result image with boxes and labels on it.
		image_np = load_image_into_numpy_array(image.convert('RGB'))
		# Expand dimensions since the model expects images to have shape: [1, None, None, 3]
		image_np_expanded = np.expand_dims(image_np, axis=0)
		# Actual detection.
		print("inside detection method")
		output_dict = run_inference_for_single_image(image_np, detection_graph)
		license_plate_characters = self.getValues(output_dict,category_index)
		return license_plate_characters;

		
		
