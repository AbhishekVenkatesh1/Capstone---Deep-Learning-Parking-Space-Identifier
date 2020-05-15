import numpy as np
import cv2
from matplotlib import pyplot as plt
from math import sqrt
import os
from PIL import Image
from keras.applications.imagenet_utils import preprocess_input
from keras.models import load_model
from keras.preprocessing import image
from os import listdir
from os.path import isfile, join


cwd = os.getcwd()
model_weights = 'car_model.h5'

output_dict = {}
output_dict[0] = 'empty'
output_dict[1] = 'occupied'

model = load_model(model_weights)

class Parking_Lot:

	def __init__(self, filename):
		self.filename = filename

	# def __init__(self):
	# 	pass

	# Canny transform to detect edges of image
	def canny_transform(self,image):
		gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) # Converting image to Grayscale
		blur = cv2.GaussianBlur(gray, (5,5), 0) # Doing Gaussian Blur to image
		canny = cv2.Canny(blur, 50, 150) # Doing Canny edge transform
		return canny

	# Croppping image to just include parking lot and nothing else
	def region_of_interest(self,image):
		height = image.shape[0]
		width = image.shape[1]
		# coordinates of region of interest - CHANGE BASED ON IMAGE
		polygons = np.array([[]])
		if self.filename == '/Users/abhishek.venkatesh/Desktop/Capstone/Data/model_images/Parking Lot 1.jpg':
			polygons = np.array([[(0,490), (1200,490), (1100,41), (150,41), (0,250)]]) # img1
			# print(polygons)
		elif self.filename == '/Users/abhishek.venkatesh/Desktop/Capstone/Data/model_images/Parking Lot 2.jpg':
			polygons = np.array([[(0,45), (433,45), (433,372), (0,372)]]) # img2
			# print(polygons)
		else:
			polygons = np.array([[(0,36), (653,36), (653,355), (0,355)]]) # img3
			# print(polygons)
		# polygons = np.array([[(0,490), (1200,490), (1100,41), (150,41), (0,250)]]) # img1
		# polygons = np.array([[(0,45), (433,45), (433,372), (0,372)]]) # img2
		# polygons = np.array([[(0,36), (653,36), (653,355), (0,355)]]) # img3

		mask = np.zeros_like(image) # black the image
		cv2.fillPoly(mask, polygons, 255) # region of interest is colored in white
		masked_image = cv2.bitwise_and(image,mask) # combined masked area with original image
		return masked_image

	# Display array of lines in an image
	def display_lines_old(self,image, lines):
		line_image = np.zeros_like(image) # Black the image
		if lines is not None:
			for line in lines:
				x1, y1, x2, y2 = line.reshape(4)
				cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
		return line_image

	# Display array of lines in an image
	def display_lines(self,image, lines):
		line_image = np.zeros_like(image) # Black the image
		if lines is not None:
			for line in lines:
				x1 = int(line[0])
				y1 = int(line[1])
				x2 = int(line[2])
				y2 = int(line[3])
				cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
		return line_image

	# Find the average lines of lines that are close to each other
	def average_lines(self,lines):
		new_space_lines = [] # new space lines of slope between 3 and 10 or straight vertical
		new_space_lines_map = {} # map to check if line has been used to calculate average yet
		averaged_space_lines = [] 


		slope_min = 3 
		slope_max = 10 
		# Only finding lines with slope between slope_min and slope_max or straight vertical
		for i in range(len(lines)):
			# calculating difference between x2 and x1 values and y2 and y1 values
			if lines[i][0][2]-lines[i][0][0]==0 or lines[i][0][3]-lines[i][0][1]==0:
				new_space_lines.append(lines[i][0])
				continue
			# checking for slope
			# CHANGE BASED ON IMAGE: img1 - need, img2/img3 - comment out
			# if slope_min <= abs((lines[i][0][3]-lines[i][0][1])/(lines[i][0][2]-lines[i][0][0])) <= slope_max:
			if self.filename == '/Users/abhishek.venkatesh/Desktop/Capstone/Data/model_images/Parking Lot 1.jpg':
				new_space_lines.append(lines[i][0])

		# Setting map values to True for all new space lines
		for i in range(len(new_space_lines)):
			new_space_lines_map[i] = True

		x_dist_val = 5 
		y_dist_val = 20 
		# Nested loop to check every combination of two lines
		for i in range(len(new_space_lines)):
			for j in range(len(new_space_lines)):
				if i != j:
					# Checking if a line has not been used already to determine an average line
					if new_space_lines_map[i] is True and new_space_lines_map[j] is True:
						x_dist = abs(new_space_lines[i][0]-new_space_lines[j][0])
						y_dist = abs(new_space_lines[i][1]-new_space_lines[j][1])
						# Only calculating average of lines if they are within 5 in x values and 20 in y values
						if x_dist <= x_dist_val and y_dist <= y_dist_val:
							# if new_space_lines[i] is True and new_space_lines_map[j] is True:
							new_space_lines_map[i] = False
							new_space_lines_map[j] = False
							x1_avg = (new_space_lines[i][0]+new_space_lines[j][0])/2
							y1_avg = (new_space_lines[i][1]+new_space_lines[j][1])/2
							x2_avg = (new_space_lines[i][2]+new_space_lines[j][2])/2
							y2_avg = (new_space_lines[i][3]+new_space_lines[j][3])/2
							averaged_space_lines.append((x1_avg, y1_avg, x2_avg, y2_avg))

		# Appending extra lines to map that are not close to any other lines which means that there wasn't any average calculated for them
		for i, value in new_space_lines_map.items():
			if value is True:
				x1 = new_space_lines[i][0]
				y1 = new_space_lines[i][1]
				x2 = new_space_lines[i][2]
				y2 = new_space_lines[i][3]
				averaged_space_lines.append((x1, y1, x2, y2))

		return averaged_space_lines

	# identify rectangular blocks of lines
	def draw_blocks(self,image, lines):
		new_image = np.copy(image)

		# appending lines into a new array of format (y1, x1, y2, x2)
		cleaned = []
		for i in range(len(lines)):
			# line in format (y1, x1, y2, x2) because want to sort by y1
			line = (lines[i][1], lines[i][0], lines[i][3], lines[i][2])
			cleaned.append(line)

		# Appending lines to various clusters
		import operator
		sorted_list = sorted(cleaned, key=operator.itemgetter(0,1))
		clusters = {} # each key is a cluster and the values are the parking space in each cluster
		dIndex = 0
		# CHANGE BASED ON IMAGE
		clus_y_dist = 0
		if self.filename == '/Users/abhishek.venkatesh/Desktop/Capstone/Data/model_images/Parking Lot 2.jpg':
			clus_y_dist = 20
		elif self.filename == '/Users/abhishek.venkatesh/Desktop/Capstone/Data/model_images/Parking Lot 3.jpg':
			clus_y_dist = 5
		else:
			clus_y_dist = 2
		# clus_y_dist = 5 # max distance between lines in cluster, img1 = 5, img2 = 20, img3 = 5
		for i in range(len(sorted_list)-1):
			y_dist = abs(sorted_list[i+1][0]-sorted_list[i][0])
			if y_dist <= clus_y_dist:
				if not dIndex in clusters.keys():
					clusters[dIndex] = []
				if sorted_list[i] not in clusters[dIndex]:
					clusters[dIndex].append(sorted_list[i])
				if sorted_list[i+1] not in clusters[dIndex]:
					clusters[dIndex].append(sorted_list[i+1])
			else:
				dIndex += 1

		i=0
		polygons = {} # 4 coordinates of each cluster to draw rectangle
		for key in clusters:
			all_list = clusters[key]
			cleaned = list(set(all_list))
			# if len(cleaned) > 5:
			# create 4 different cleaned lists that are sorted by x1, y1, x2, and y2
			cleaned_x1 = sorted(cleaned, key=lambda tup: tup[1])
			cleaned_y1 = sorted(cleaned, key=lambda tup: tup[0])
			cleaned_x2 = sorted(cleaned, key=lambda tup: tup[3])
			cleaned_y2 = sorted(cleaned, key=lambda tup: tup[2])

			# BOTTOM LEFT - (smallest x1, biggest y1) -> (min_x1, max_y1)
			min_x1 = cleaned_x1[0][1]
			max_y1 = cleaned_y1[-1][0]
			b_left = [min_x1, max_y1]
			# TOP RIGHT - (biggest x2, smallest y2) -> (max_x2, min_y2)
			max_x2 = cleaned_x2[-1][3]
			min_y2 = cleaned_y2[0][2]
			t_right = [max_x2, min_y2]
			
			# TOP LEFT - (smallest x2, smallest y2) -> (min_x2, min_y2)
			min_x2 = cleaned_x2[0][3]
			min_y2 = cleaned_y2[0][2]
			t_left = [min_x2, min_y2]
			# BOTTOM RIGHT - (biggest x1, biggest y1) -> (max_x1, max_y1)
			max_x1 = cleaned_x1[-1][1]
			max_y1 = cleaned_y1[-1][0]
			b_right = [max_x1, max_y1]
			
			# Putting array coordiantes of polygon edges into polygons map
			polygons[key] = [b_left, t_left, t_right, b_right]
			i += 1

			# Drawing the rectangular boxes on image
			pts = np.array([b_left, t_left, t_right, b_right], np.int32)
			cv2.polylines(new_image, [pts], True, (0,255,0), 2)

		# final_polygons = {}
		# buff = 0
		# j=0
		# for key in polygons:
		# 	# Applying 5 unit buffer to the coordinates of the polygon
		# 	bottom_left_x = polygons[key][0][0]-buff
		# 	top_left_x = polygons[key][1][0]-buff
		# 	top_right_x = polygons[key][2][0]+buff
		# 	bottom_right_x = polygons[key][3][0]+buff
		# 	bottom_left_y = polygons[key][0][1]+buff
		# 	top_left_y = polygons[key][1][1]-buff
		# 	top_right_y = polygons[key][2][1]-buff
		# 	bottom_right_y = polygons[key][3][1]+buff

		# 	# Creating new polygons coordinates with the buffer
		# 	bottom_left = [bottom_left_x, bottom_left_y]
		# 	top_left = [top_left_x, top_left_y]
		# 	top_right = [top_right_x, top_right_y]
		# 	bottom_right = [bottom_right_x, bottom_right_y]

		# 	# Appending coordinates of new polygon into final_polygons array
		# 	final_polygons[key] = [bottom_left, top_left, top_right, bottom_right]
		# 	j += 1

		# 	# Drawing the rectangular boxes on image
		# 	pts = np.array([bottom_left, top_left, top_right, bottom_right], np.int32)
		# 	cv2.polylines(new_image, [pts], True, (0,255,0), 2)

		return new_image, clusters, polygons

	def create_parking(self,image, clusters, polygons):
		new_image = np.copy(image)

		# clusters has list of lines in each cluster number in format (y1, x1, y2, x2)
		# polygons has coordinates 4 points of polygons that represent sections of parking lot
		# clusters_new rearranges lines in format (x1, y2, x2, y2)
		clusters_new = {}
		for key in clusters:
			lines = clusters[key]
			clusters_new[key] = []
			for line in lines:
				x1 = line[1]
				y1 = line[0]
				x2 = line[3]
				y2 = line[2]
				new_line = (x1, y1, x2, y2)
				clusters_new[key].append(new_line)

		# middle_of_blocks_lines store middle lines that divide blocks into two halves
		# spot_divider_lines are the actual parking spot lines
		middle_of_blocks_lines = {}
		spot_divider_lines = {}
		for key in polygons:
			bottom_left = polygons[key][0]
			top_left = polygons[key][1]
			top_right = polygons[key][2]
			bottom_right = polygons[key][3]

			bottom_left_x = bottom_left[0]
			top_left_x = top_left[0]
			top_right_x = top_right[0]
			bottom_right_x = bottom_right[0]

			bottom_left_y = bottom_left[1]
			top_left_y = top_left[1]
			top_right_y = top_right[1]
			bottom_right_y = bottom_right[1]

			# Determining bottom of polygon by checking if left side or right side of block is lower
			polygon_bottom_y = 0
			if bottom_left_y > bottom_right_y:
				polygon_bottom_y = bottom_left_y
			else:
				polygon_bottom_y = bottom_right_y

			# Determining top of polygon by checking if left side or right side is higher
			polygon_top_y = 0
			if top_left_y < top_right_y:
				polygon_top_y = top_left_y
			else:
				polygon_top_y = top_right_y

			# Averaging out top and bottom coordinates of polygons to find coordinates of middle
			# middle_of_blocks_lines stores cooridnates of two points that make up middle line
			polygon_middle_left_x = 0
			polygon_middle_left_y = 0
			polygon_middle_right_x = 0
			polygon_middle_right_y = 0
			# CHANGE BASED ON IMAGE
			# if top_left_y < 100000000: # img1 = 400, img2 = very large b/c not needed, img4 = very large, 
			if self.filename == '/Users/abhishek.venkatesh/Desktop/Capstone/Data/model_images/Parking Lot 1.jpg':
				if top_left_y < 400:
					polygon_middle_left_x = int((top_left_x + bottom_left_x)/2)
					polygon_middle_left_y = int((top_left_y + bottom_left_y)/2)
					polygon_middle_right_x = int((top_right_x + bottom_right_x)/2)
					polygon_middle_right_y = int((top_right_y + bottom_right_y)/2)
					middle_of_blocks_lines[key] = (polygon_middle_left_x, polygon_middle_left_y, polygon_middle_right_x, polygon_middle_right_y)
					cv2.line(new_image, (polygon_middle_left_x, polygon_middle_left_y), (polygon_middle_right_x, polygon_middle_right_y), (0, 255, 0), 2)
			else:
				polygon_middle_left_x = int((top_left_x + bottom_left_x)/2)
				polygon_middle_left_y = int((top_left_y + bottom_left_y)/2)
				polygon_middle_right_x = int((top_right_x + bottom_right_x)/2)
				polygon_middle_right_y = int((top_right_y + bottom_right_y)/2)
				middle_of_blocks_lines[key] = (polygon_middle_left_x, polygon_middle_left_y, polygon_middle_right_x, polygon_middle_right_y)
				cv2.line(new_image, (polygon_middle_left_x, polygon_middle_left_y), (polygon_middle_right_x, polygon_middle_right_y), (0, 255, 0), 2)


			# spot_divider_lines[key][0] is bottom of block
			# spot_divider_lines[key][1] is top of block
			spot_divider_lines[key] = []
			spot_divider_lines[key].append([])
			spot_divider_lines[key].append([])
			cluster_lines = clusters_new[key]
			for line in cluster_lines:
				x1 = line[0]
				y1 = line[1]
				x2 = line[2]
				y2 = line[3]

				slope = 0
				is_slope_zero = False
				if x2 == x1 or y2 == y1:
					is_slope_zero = True
				else:
					slope = (y2-y1)/(x2-x1)

				# determining the bottom and top of lines
				line_bottom_y = 0
				line_top_y = 0
				line_bottom_x = 0
				line_top_x = 0
				if y1 > y2:
					line_bottom_y = y1
					line_top_y = y2
					line_bottom_x = x1
					line_top_x = x2
				else:
					line_bottom_y = y2
					line_top_y = y1
					line_bottom_x = x2
					line_top_x = x1


				# Calculating how much distance to add to initial hough lines to cover the whole block
				# calculating bottom point of parking spot line
				# point will be (final_line_bottom_x, final_line_bottom_y)
				final_line_bottom_x = 0
				final_line_bottom_y = 0
				# is_bottom = False
				if line_bottom_y > polygon_middle_right_y: # lower half of block
					# is_bottom = True
					difference = polygon_bottom_y - line_bottom_y
					if is_slope_zero is False and difference > 0:
						x_bottom_change = abs(difference/slope)
						if slope < 0:
							final_line_bottom_x = line_bottom_x - x_bottom_change
						else:
							final_line_bottom_x = line_bottom_x + x_bottom_change
					else:
						final_line_bottom_x = line_bottom_x
					final_line_bottom_y = polygon_bottom_y
				else: # upper half of block
					difference = polygon_middle_right_y - line_bottom_y
					if is_slope_zero is False and difference > 0:
						x_bottom_change = abs(difference/slope)
						if slope < 0:
							final_line_bottom_x = line_bottom_x - x_bottom_change
						else:
							final_line_bottom_x = line_bottom_x + x_bottom_change
					else:
						final_line_bottom_x = line_bottom_x
					final_line_bottom_y = polygon_middle_right_y


				# calculating top point of parking spot line
				# point will be (final_line_top_x, final_line_top_y)
				final_line_top_x = 0
				final_line_top_y = 0
				# is_top = False
				if line_top_y > polygon_middle_right_y and polygon_middle_right_y != 0: # lower half of block when there's no upper half
					# is_top = True
					# if is_bottom == True and is_top == True:
					# 	final_line_top_x = polygon_middle_right_x
					# 	final_line_top_y = polygon_middle_right_y
					difference = polygon_middle_right_y - line_bottom_y # MIGHT HAVE TO COMMENT OUT
					if is_slope_zero is False and difference < 0:
						x_top_change = abs(difference/slope)
						if slope < 0:
							final_line_top_x = line_top_x + x_top_change
						else:
							final_line_top_x = line_top_x - x_top_change
					else:
						final_line_top_x = line_top_x
					final_line_top_y = polygon_middle_right_y
					spot_divider_lines[key][0].append((final_line_bottom_x, final_line_bottom_y, final_line_top_x, final_line_top_y))
				else: # upper half of block
					difference = polygon_top_y - line_top_y # negative number
					if is_slope_zero is False and difference < 0:
						x_top_change = abs(difference/slope)
						if slope < 0:
							final_line_top_x = line_top_x + x_top_change
						else:
							final_line_top_x = line_top_x - x_top_change
					else:
						final_line_top_x = line_top_x
					final_line_top_y = polygon_top_y
					if polygon_middle_right_y != 0: # upper half of block
						spot_divider_lines[key][1].append((final_line_bottom_x, final_line_bottom_y, final_line_top_x, final_line_top_y))
					else: # lower half of block when there's no upper half
						spot_divider_lines[key][0].append((final_line_bottom_x, final_line_bottom_y, final_line_top_x, final_line_top_y))

				cv2.line(new_image, (int(final_line_bottom_x), int(final_line_bottom_y)), (int(final_line_top_x), int(final_line_top_y)), (0, 255, 0), 2)

		return new_image, middle_of_blocks_lines, spot_divider_lines, clusters_new

	def create_final_parking(self, image, cluster_polygons, middle_block_lines, spot_divider_lines):
		new_image = np.copy(image)

		# drawing blocks of parking spots
		for key in cluster_polygons:
			bottom_left = cluster_polygons[key][0]
			top_left = cluster_polygons[key][1]
			top_right = cluster_polygons[key][2]
			bottom_right = cluster_polygons[key][3]
			pts = np.array([bottom_left, top_left, top_right, bottom_right], np.int32)
			cv2.polylines(new_image, [pts], True, (0,255,0), 2)

		# drawing middle of block lines if exists
		for key in middle_block_lines:
			x1 = middle_block_lines[key][0]
			y1 = middle_block_lines[key][1]
			x2 = middle_block_lines[key][2]
			y2 = middle_block_lines[key][3]
			cv2.line(new_image, (x1, y1), (x2, y2), (0, 255, 0), 2)


		# Determining final spot lines for parking spots
		final_lines = {}
		for key in spot_divider_lines:
			final_lines[key] = []
			final_lines[key].append([])
			final_lines[key].append([])
			bottom_lines = spot_divider_lines[key][0]
			top_lines = spot_divider_lines[key][1]
			sorted_bottom_lines = sorted(bottom_lines, key=lambda tup: tup[0]) # sorted bottom lines up x1
			sorted_top_lines = []
			if len(top_lines) > 0:
				sorted_top_lines = sorted(top_lines, key=lambda tup: tup[0]) # sorted top lines by x1

			bottom_lines_taken = {}
			top_lines_taken = {}
			for i in range(len(sorted_bottom_lines)):
				bottom_lines_taken[i] = 0
			for i in range(len(sorted_top_lines)):
				top_lines_taken[i] = 0

			bottom_left_x = cluster_polygons[key][0][0]
			bottom_left_y = cluster_polygons[key][0][1]
			top_left_x = cluster_polygons[key][1][0]
			top_left_y = cluster_polygons[key][1][1]
			top_right_x = cluster_polygons[key][2][0]
			top_right_y = cluster_polygons[key][2][1]
			bottom_right_x = cluster_polygons[key][3][0]
			bottom_right_y = cluster_polygons[key][3][1]

			# CHANGE BASED ON IMAGE
			dist = 0
			x_dist_min = 0
			x_dist_max = 0
			if self.filename == '/Users/abhishek.venkatesh/Desktop/Capstone/Data/model_images/Parking Lot 1.jpg':
				dist = 35
				x_dist_min = 35
				x_dist_max = 55
			else:
				dist = 25
				x_dist_min = 25
				x_dist_max = 35
			# dist = 25 # img1 = 35, img2/img3 = 25
			# x_dist_min = 25 # img1 = 35, img2/img3 = 25
			# x_dist_max = 35 # img1 = 55, img2/img3 = 35
			# map to say if bottom line is valid to be in final lines
			sorted_bottom_valid = {}
			for i in range(len(sorted_bottom_lines)):
				sorted_bottom_valid[i] = True # initially all lines are valid to be in final lines

			for i in range(len(sorted_bottom_lines)):
				x_dist_from_left_block_edge = abs(sorted_bottom_lines[i][0] - bottom_left_x)
				if x_dist_from_left_block_edge < dist: # too close to edge to be a parking spot line
					sorted_bottom_valid[i] = False

				for j in range(len(sorted_bottom_lines)):
					x_dist_from_right_block_edge = abs(bottom_right_x - sorted_bottom_lines[j][0])
					if x_dist_from_right_block_edge < dist: # too close to edge to be a parking spot line
						sorted_bottom_valid[j] = False
					if j > i and sorted_bottom_valid[i] is True:
						x_dist = abs(sorted_bottom_lines[j][0] - sorted_bottom_lines[i][0])
						if x_dist_min <= x_dist <= x_dist_max: # only looking for lines that are between certain pixels away
							# line will be used in final lines when it is the left of two lines and/or right of two lines so max of used 2 times
							if bottom_lines_taken[i] < 2 and bottom_lines_taken[j] < 2: # lines calulating in left to right fashion
									bottom_lines_taken[i] += 1
									bottom_lines_taken[j] += 1
									if sorted_bottom_lines[i] not in final_lines[key][0]:
										final_lines[key][0].append(sorted_bottom_lines[i])
									if sorted_bottom_lines[j] not in final_lines[key][0] and sorted_bottom_valid[j] is True:
										final_lines[key][0].append(sorted_bottom_lines[j])
						if x_dist < dist:
							sorted_bottom_valid[j] = False

			# map to say it top line is valid to be in final lines
			sorted_top_valid = {}
			for i in range(len(sorted_top_lines)):
				sorted_top_valid[i] = True # initially all lines are valid to be in final lines

			for i in range(len(sorted_top_lines)):
				x_dist_from_left_block_edge = abs(sorted_top_lines[i][0] - top_left_x)
				if x_dist_from_left_block_edge < dist: # too close to edge to be a parking spot line
					sorted_top_valid[i] = False
				for j in range(len(sorted_top_lines)):
					x_dist_from_right_block_edge = abs(top_right_x - sorted_top_lines[j][0])
					if x_dist_from_right_block_edge < dist: # too close to edge to be a parking spot line
						sorted_top_valid[j] = False
					if j > i and sorted_top_valid[i] is True:
						x_dist = abs(sorted_top_lines[j][0] - sorted_top_lines[i][0])
						if x_dist_min <= x_dist <= x_dist_max: # only looking for lines that are between certain pixels away
							# line will be used in final lines when it is the left of two lines and/or right of two lines so max of used 2 times
							if top_lines_taken[i] < 2 and top_lines_taken[j] < 2: # lines calulating in left to right fashion
									top_lines_taken[i] += 1
									top_lines_taken[j] += 1
									if sorted_top_lines[i] not in final_lines[key][1]:
										final_lines[key][1].append(sorted_top_lines[i])
									if sorted_top_lines[j] not in final_lines[key][1] and sorted_top_valid[j] is True:
										final_lines[key][1].append(sorted_top_lines[j])
						if x_dist < dist: 
							sorted_top_valid[j] = False

		# adding missing spot lines to map
		new_final_lines = {}
		for key in final_lines:
			new_final_lines[key] = []
			new_final_lines[key].append([])
			new_final_lines[key].append([])

			bottom_left = cluster_polygons[key][0]
			top_left = cluster_polygons[key][1]
			top_right = cluster_polygons[key][2]
			bottom_right = cluster_polygons[key][3]

			bottom_left_x = bottom_left[0]
			bottom_left_y = bottom_left[1]
			top_left_x = top_left[0]
			top_left_y = top_left[1]
			top_right_x = top_right[0]
			top_right_y = top_right[1]
			bottom_right_x = bottom_right[0]
			bottom_right_y = bottom_right[1]

			# checking for additional spots in bottom half of block
			bottom_lines = final_lines[key][0]
			sorted_bottom_lines = sorted(bottom_lines, key=lambda tup: tup[0])
			count = 0
			prev_x = 0
			for line in sorted_bottom_lines:
				x1 = int(line[0])
				y1 = int(line[1])
				x2 = int(line[2])
				y2 = int(line[3])
				x_copy = 0
				if (x1 > x2):
					x_copy = x1
				else:
					x_copy = x2
				if abs(x2-x1) > 15: 
					continue
				new_final_lines[key][0].append((x1, y1, x2, y2))
				# cv2.line(new_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

				# CHANGE BASED ON IMAGE
				spot_length = 0
				if self.filename == '/Users/abhishek.venkatesh/Desktop/Capstone/Data/model_images/Parking Lot 1.jpg':
					spot_length = 40
				else:
					spot_length = 30
				# spot_length = 30 # img1 = 40, img2/img3 = 30
				# since spots are certain pixels wide, checking it there's space for spots to the left of the first spot line of row in map
				if count == 0:
					x_dist = x1 - bottom_left_x
					x1 -= spot_length
					x2 -= spot_length
					# looping to check for all the potential spots to the left of first line
					while x_dist > 0:
						new_final_lines[key][0].append((x1, y1, x2, y2))
						# cv2.line(new_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
						x1 -= spot_length
						x2 -= spot_length
						x_dist = x1 - bottom_left_x
					count += 1
				# checking to see if there's space for spots to the right of the last line of row in map
				elif count == len(sorted_bottom_lines)-1:
					x_dist = bottom_right_x - x1
					x1 += spot_length
					x2 += spot_length
					# loop to check for potential spot lines to the right of last line
					while x_dist > 0:
						new_final_lines[key][0].append((x1, y1, x2, y2))
						# cv2.line(new_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
						x1 += spot_length
						x2 += spot_length
						x_dist = bottom_right_x - x1
				# checking to see if there's space for lines ot the left of current line and to the right of previous line
				else:
					x_dist = x1 - prev_x
					x1 -= spot_length
					x2 -= spot_length
					while x_dist > spot_length:
						new_final_lines[key][0].append((x1, y1, x2, y2))
						# cv2.line(new_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
						x1 -= spot_length
						x2 -= spot_length
						x_dist = x1 - prev_x
					count += 1
				prev_x = x_copy

			# checking for additional spots in top half of block
			top_lines = final_lines[key][1]
			sorted_top_lines = sorted(top_lines, key=lambda tup: tup[0])
			count = 0
			for line in sorted_top_lines:
				x1 = int(line[0])
				y1 = int(line[1])
				x2 = int(line[2])
				y2 = int(line[3])
				x_copy = 0
				if (x1 > x2):
					x_copy = x1
				else:
					x_copy = x2
				if abs(x2-x1) > 15:
					continue
				new_final_lines[key][1].append((x1, y1, x2, y2))
				# cv2.line(new_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

				# CHANGE BASED ON IMAGE
				spot_length = 0
				if self.filename == '/Users/abhishek.venkatesh/Desktop/Capstone/Data/model_images/Parking Lot 1.jpg':
					spot_length = 40
				else:
					spot_length = 30
				# spot_length = 30 # img1 = 40, img2/img3 = 30
				# since spots are certain  pixels wide, checking it there's space for spots to the left of the first spot line of row in map
				if count == 0:
					x_dist = x1 - top_left_x
					x1 -= spot_length
					x2 -= spot_length
					# looping to check for all the potential spots to the left of first line
					while x_dist > 0:
						new_final_lines[key][1].append((x1, y1, x2, y2))
						# cv2.line(new_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
						x1 -= spot_length
						x2 -= spot_length
						x_dist = x1 - top_left_x
					count += 1
				# checking to see if there's space for spots to the right of the last line of row in map
				elif count == len(sorted_top_lines)-1:
					x_dist = top_right_x - x1
					x1 += spot_length
					x2 += spot_length
					# loop to check for potential spot lines to the right of last line
					while x_dist > 0:
						new_final_lines[key][1].append((x1, y1, x2, y2))
						# cv2.line(new_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
						x1 += spot_length
						x2 += spot_length
						x_dist = top_right_x - x1
				# checking to see if there's space for lines ot the left of current line and to the right of previous line
				else:
					x_dist = x1 - prev_x
					x1 -= spot_length
					x2 -= spot_length
					while x_dist > 0:
						new_final_lines[key][1].append((x1, y1, x2, y2))
						# cv2.line(new_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
						x1 -= spot_length
						x2 -= spot_length
						x_dist = x1 - prev_x
					count += 1
				prev_x = x_copy

		# drawing these final lines
		for key in new_final_lines:
			sort_b = sorted(new_final_lines[key][0], key=lambda tup: tup[0])
			sort_t = sorted(new_final_lines[key][1], key=lambda tup: tup[0])

			for i in range(len(sort_b)-1):
				if sort_b[i+1][0]-sort_b[i][0] < 20:
					new_final_lines[key][0].remove(sort_b[i])

			for i in range(len(sort_t)-1):
				if sort_t[i+1][0]-sort_t[i][0] < 20:
					new_final_lines[key][1].remove(sort_t[i])

			# drawing these final lines
			for line in new_final_lines[key][0]:
				x1 = int(line[0])
				y1 = int(line[1])
				x2 = int(line[2])
				y2 = int(line[3])
				cv2.line(new_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
			for line in new_final_lines[key][1]:
				x1 = int(line[0])
				y1 = int(line[1])
				x2 = int(line[2])
				y2 = int(line[3])
				cv2.line(new_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

		return new_image, new_final_lines

	# creating a map of all the spots and assigning them a number
	def create_spots_ids(self,polygons, middle_lines, spot_lines):
		spots_map = {} # map of all spots
		count = 0 # total number of spots
		count_map = {} # mapping number of spots in block to the block number

		for key in polygons:
			curr_count = 0 # number of spots in given block

			bottom_left_x = polygons[key][0][0]
			bottom_left_y = polygons[key][0][1]
			top_left_x = polygons[key][1][0]
			top_left_y = polygons[key][1][1]
			top_right_x = polygons[key][2][0]
			top_right_y = polygons[key][2][1]
			bottom_right_x = polygons[key][3][0]
			bottom_right_y = polygons[key][3][1]

			middle_left_x = 0
			middle_left_y = 0
			middle_right_x = 0
			middle_right_y = 0
			# checking for if block has have two halves annd assigning values to the coordinates of the middle line
			# if block doesn't have two halves, then middle line is is top of block
			if key in middle_lines.keys():
				middle_x1 = middle_lines[key][0]
				middle_y1 = middle_lines[key][1]
				middle_x2 = middle_lines[key][2]
				middle_y2 = middle_lines[key][3]
				if middle_x1 < middle_y1:
					middle_left_x = middle_x1
					middle_left_y = middle_y1
					middle_right_x = middle_x2
					middle_right_y = middle_y2
				else:
					middle_left_x = middle_x2
					middle_left_y = middle_y2
					middle_right_x = middle_x1
					middle_right_y = middle_y1
			else:
				middle_left_x = top_left_x
				middle_left_y = top_left_y
				middle_right_x = top_right_x
				middle_right_y = top_right_y

			# sort the bottom and top halfs of lines
			bottom_spot_lines = sorted(spot_lines[key][0], key=lambda tup: tup[0])
			top_spot_lines = sorted(spot_lines[key][1], key=lambda tup: tup[0])

			# finding coordinates of spots in bottom half of row
			if len(bottom_spot_lines) > 0:
				for i in range(len(bottom_spot_lines)-1):
					# a given line
					line1_x1 = bottom_spot_lines[i][0]
					line1_y1 = bottom_spot_lines[i][1]
					line1_x2 = bottom_spot_lines[i][2]
					line1_y2 = bottom_spot_lines[i][3]

					# the line to the right of given line
					j = i+1
					line2_x1 = bottom_spot_lines[j][0]
					line2_y1 = bottom_spot_lines[j][1]
					line2_x2 = bottom_spot_lines[j][2]
					line2_y2 = bottom_spot_lines[j][3]

					# calcluating values for botton and top of a certain line
					line1_top_x = 0
					line1_top_y = 0
					line1_bottom_x = 0
					line1_bottom_y = 0
					if line1_y1 > line1_y2:
						line1_top_x = line1_x2
						line1_top_y = line1_y2
						line1_bottom_x = line1_x1
						line1_bottom_y = line1_y1 
					else:
						line1_top_x = line1_x1
						line1_top_y = line1_y1
						line1_bottom_x = line1_x2
						line1_bottom_y = line1_y2

					# calcluating values for botton and top of a certain line
					line2_top_x = 0
					line2_top_y = 0
					line2_bottom_x = 0
					line2_bottom_y = 0
					if line2_y1 > line2_y2:
						line2_top_x = line2_x2
						line2_top_y = line2_y2
						line2_bottom_x = line2_x1
						line2_bottom_y = line2_y1 
					else:
						line2_top_x = line2_x1
						line2_top_y = line2_y1
						line2_bottom_x = line2_x2
						line2_bottom_y = line2_y2

					# finding 4 coordinates of the first spot in a row which will be 4 sided polygon
					if i == 0:
						spot_b_left = (bottom_left_x, bottom_left_y)
						spot_t_left = (middle_left_x, middle_left_y)
						spot_t_right = (line1_top_x, line1_top_y)
						spot_b_right = (line1_bottom_x, line1_bottom_y)
						spots_map[count] = (spot_b_left, spot_t_left, spot_t_right, spot_b_right)
						count += 1
						curr_count += 1
					# finding 4 coordinates of any subsequent spot but the last spot in row which will be 4 sided polygon
					else:
						spot_b_left = (line1_bottom_x, line1_bottom_y)
						spot_t_left = (line1_top_x, line1_top_y)
						spot_t_right = (line2_top_x, line2_top_y)
						spot_b_right = (line2_bottom_x, line2_bottom_y)
						spots_map[count] = (spot_b_left, spot_t_left, spot_t_right, spot_b_right)
						count += 1
						curr_count += 1

				# finding 4 coordinates of last spot in row which will be 4 sided polygon
				i = len(bottom_spot_lines)-1
				line_x1 = bottom_spot_lines[i][0]
				line_y1 = bottom_spot_lines[i][1]
				line_x2 = bottom_spot_lines[i][2]
				line_y2 = bottom_spot_lines[i][3]
				if line_y1 > line_y2:
					line_top_x = line_x2
					line_top_y = line_y2
					line_bottom_x = line_x1
					line_bottom_y = line_y1 
				else:
					line_top_x = line_x1
					line_top_y = line_y1
					line_bottom_x = line_x2
					line_bottom_y = line_y2
				spot_b_left = (line_bottom_x, line_bottom_y)
				spot_t_left = (line_top_x, line_top_y)
				spot_t_right = (middle_right_x, middle_right_y)
				spot_b_right = (bottom_right_x, bottom_right_y)
				spots_map[count] = (spot_b_left, spot_t_left, spot_t_right, spot_b_right)
				count += 1
				curr_count += 1

			# finding coordinates of spots in top half of row
			if len(top_spot_lines) > 0:
				for i in range(len(top_spot_lines)-1):
					# a given line
					line1_x1 = top_spot_lines[i][0]
					line1_y1 = top_spot_lines[i][1]
					line1_x2 = top_spot_lines[i][2]
					line1_y2 = top_spot_lines[i][3]

					# the line to the right of given line
					j = i+1
					line2_x1 = top_spot_lines[j][0]
					line2_y1 = top_spot_lines[j][1]
					line2_x2 = top_spot_lines[j][2]
					line2_y2 = top_spot_lines[j][3]

					# calcluating values for botton and top of a certain line
					line1_top_x = 0
					line1_top_y = 0
					line1_bottom_x = 0
					line1_bottom_y = 0
					if line1_y1 > line1_y2:
						line1_top_x = line1_x2
						line1_top_y = line1_y2
						line1_bottom_x = line1_x1
						line1_bottom_y = line1_y1 
					else:
						line1_top_x = line1_x1
						line1_top_y = line1_y1
						line1_bottom_x = line1_x2
						line1_bottom_y = line1_y2

					line2_top_x = 0
					line2_top_y = 0
					line2_bottom_x = 0
					line2_bottom_y = 0
					if line2_y1 > line2_y2:
						line2_top_x = line2_x2
						line2_top_y = line2_y2
						line2_bottom_x = line2_x1
						line2_bottom_y = line2_y1 
					else:
						line2_top_x = line2_x1
						line2_top_y = line2_y1
						line2_bottom_x = line2_x2
						line2_bottom_y = line2_y2

					# finding 4 coordinates of the first spot in a row which will be 4 sided polygon
					if i == 0:
						spot_b_left = (middle_left_x, middle_left_y)
						spot_t_left = (top_left_x, top_left_y)
						spot_t_right = (line1_top_x, line1_top_y)
						spot_b_right = (line1_bottom_x, line1_bottom_y)
						spots_map[count] = (spot_b_left, spot_t_left, spot_t_right, spot_b_right)
						count += 1
						curr_count += 1
					# finding 4 coordinates of any subsequent spot but the last spot in row which will be 4 sided polygon
					else:
						spot_b_left = (line1_bottom_x, line1_bottom_y)
						spot_t_left = (line1_top_x, line1_top_y)
						spot_t_right = (line2_top_x, line2_top_y)
						spot_b_right = (line2_bottom_x, line2_bottom_y)
						spots_map[count] = (spot_b_left, spot_t_left, spot_t_right, spot_b_right)
						count += 1
						curr_count += 1

				# finding 4 coordinates of last spot in row which will be 4 sided polygon
				if len(bottom_spot_lines) != 0:
					i = len(bottom_spot_lines)-1
					line_x1 = bottom_spot_lines[i][0]
					line_y1 = bottom_spot_lines[i][1]
					line_x2 = bottom_spot_lines[i][2]
					line_y2 = bottom_spot_lines[i][3]
					if line_y1 > line_y2:
						line_top_x = line_x2
						line_top_y = line_y2
						line_bottom_x = line_x1
						line_bottom_y = line_y1 
					else:
						line_top_x = line_x1
						line_top_y = line_y1
						line_bottom_x = line_x2
						line_bottom_y = line_y2
					spot_b_left = (line_bottom_x, line_bottom_y)
					spot_t_left = (line_top_x, line_top_y)
					spot_t_right = (top_right_x, top_right_y)
					spot_b_right = (middle_right_x, middle_right_y)
					spots_map[count] = (spot_b_left, spot_t_left, spot_t_right, spot_b_right)
					count += 1
					curr_count += 1
			count_map[key] = curr_count


		return spots_map, count_map, count

	def draw_spot_ids(self, image, spots_map):
		new_image = np.copy(image)

		for i in range(len(spots_map)):
			spot = spots_map[i]
			spot_b_left_x = int(spot[0][0])
			spot_b_left_y = int(spot[0][1])
			spot_t_left_x = int(spot[1][0])
			spot_t_left_y = int(spot[1][1])
			spot_t_right_x = int(spot[2][0])
			spot_t_right_y = int(spot[2][1])
			spot_b_right_x = int(spot[3][0])
			spot_b_right_y = int(spot[3][1])

			spot_mid_x = int(((spot_b_left_x+spot_b_right_x)/2)-5)
			spot_mid_y = int((spot_b_left_y+spot_t_left_y)/2)

			cv2.putText(new_image, "%d" %i, (spot_mid_x, spot_mid_y),
	            cv2.FONT_HERSHEY_SIMPLEX,
	            0.5, (0, 255, 0), 2)

		return new_image


			
	def crop_spots(self, image, spots_map, dir, img_width, img_height):
		height, width = image.shape[0:2]

		path = '/Users/abhishek.venkatesh/Desktop/Capstone/Data/'+dir
		access_rights = 0o755

		try:
		    os.mkdir(path, access_rights)
		except OSError:
		    print ("Creation of the directory %s failed" % path)
		else:
		    print ("Successfully created the directory %s" % path)

		count = 1
		for i in range(len(spots_map)):
			spot = spots_map[i]
			spot_t_left_x = int(spot[1][0])
			spot_t_left_y = int(spot[1][1])
			spot_b_right_x = int(spot[3][0])
			spot_b_right_y = int(spot[3][1])

			spot_t_left_x_ratio = spot_t_left_x / width
			spot_t_left_y_ratio = spot_t_left_y / height
			spot_b_right_x_ratio = spot_b_right_x / width
			spot_b_right_y_ratio = spot_b_right_y / height

			start_row = int(height*spot_t_left_y_ratio) 
			end_row = int(height*spot_b_right_y_ratio) 
			start_col = int(width*spot_t_left_x_ratio)
			end_col = int(width*spot_b_right_x_ratio) 

			# dim = (int(img_width), int(img_height))

			cropped_img = image[start_row:end_row, start_col:end_col]
			# cropped_img = cv2.resize(cropped_img, dim)

			str_count = str(count)
			filename = 'img'+str_count+'.jpg'
			# filename1 = str_count.join(img)

			cv2.imwrite(os.path.join(path, filename), cropped_img)
			count += 1

	def predict_on_spot(self, image):
	    img = image/255.

	    image = np.expand_dims(img, axis=0)

	    class_predicted = model.predict(image) # returns numpy array of predictions
	    inID = np.argmax(class_predicted[0])
	    label = output_dict[inID]
	    return label

	def predict_on_lot(self, image, spots_map, make_copy=True, color = [0, 255, 0], alpha=0.5):
	    if make_copy:
	        new_image = np.copy(image)
	        overlay = np.copy(image)
	    num_empty = 0
	    all_spots = 0
	    empty_spots = []


	    height, width = image.shape[0:2]
	    for i in range(len(spots_map)):
	    	spot = spots_map[i]
	    	spot_b_left_x = int(spot[0][0])
	    	spot_b_left_y = int(spot[0][1])
	    	spot_t_left_x = int(spot[1][0])
	    	spot_t_left_y = int(spot[1][1])
	    	spot_t_right_x = int(spot[2][0])
	    	spot_t_right_y = int(spot[2][1])
	    	spot_b_right_x = int(spot[3][0])
	    	spot_b_right_y = int(spot[3][1])

	    	spot_b_left_x_ratio = spot_b_left_x / width
	    	spot_b_left_y_ratio = spot_b_left_y / height
	    	spot_t_left_x_ratio = spot_t_left_x / width
	    	spot_t_left_y_ratio = spot_t_left_y / height
	    	spot_t_right_x_ratio = spot_t_right_x / width
	    	spot_t_right_y_ratio = spot_t_right_y / height
	    	spot_b_right_x_ratio = spot_b_right_x / width
	    	spot_b_right_y_ratio = spot_b_right_y / height

	    	start_row = int(height*spot_t_left_y_ratio) 
	    	end_row = int(height*spot_b_right_y_ratio) 
	    	start_col = int(width*spot_t_left_x_ratio)
	    	end_col = int(width*spot_b_right_x_ratio) 

	    	spot_img = image[start_row:end_row, start_col:end_col]
	    	img_height, img_width = spot_img.shape[0:2]
	    	if img_height == 0 or img_width == 0:
	    		continue
	    	spot_img = cv2.resize(spot_img, (48, 48))

	    	label = self.predict_on_spot(spot_img)
	    	if label == 'empty':
	    		cv2.rectangle(overlay, (int(start_col),int(start_row)), (int(end_col),int(end_row)), color, -1)
	    		num_empty += 1
	    		empty_spots.append(i)
	            
	    cv2.addWeighted(overlay, alpha, new_image, 1 - alpha, 0, new_image)

	    cv2.putText(new_image, "Available: %d spots" %num_empty, (20, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 0), 2)

	    cv2.putText(new_image, "Total: %d spots" %len(spots_map), (350, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 0), 2)
	    
	    return new_image


	def execute_lots(self):
		# img1 = cv2.imread('/Users/abhishek.venkatesh/Desktop/Capstone/Data/Full_Lot_Images/2012-10-16_11_18_35.jpg')
		# img2 = cv2.imread('/Users/abhishek.venkatesh/Desktop/Capstone/Data/Full_Lot_Images/Car-Park-Space-mB_b6bb.jpg')
		# img3 = cv2.imread('/Users/abhishek.venkatesh/Desktop/Capstone/Data/Full_Lot_Images/GettyImages-873979352.jpg')
		# img4 = cv2.imread('/Users/abhishek.venkatesh/Desktop/Capstone/Data/Full_Lot_Images/scene1380.jpg')
		# img5 = cv2.imread('/Users/abhishek.venkatesh/Desktop/Capstone/Data/Full_Lot_Images/scene1410.jpg')
		# name_of_file = '/Users/abhishek.venkatesh/Desktop/Capstone/Data/model_images'+self.filename
		# img6 = cv2.imread(name_of_file)

		# edges = cv2.Canny(img3,50,150)
		# plt.subplot(121),plt.imshow(img3,cmap = 'gray')
		# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
		# plt.subplot(122),plt.imshow(edges,cmap = 'gray')
		# plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
		# plt.show()

		print(self.filename)
		img = cv2.imread(self.filename)
		parking_lot_image = np.copy(img)
		canny_edges = self.canny_transform(parking_lot_image)
		cropped_image = self.region_of_interest(canny_edges)

		space_lines = cv2.HoughLinesP(cropped_image, rho=0.1, theta=np.pi/180, threshold=5, minLineLength=5, maxLineGap=8)

		parking_space_lines_image = self.display_lines_old(parking_lot_image, space_lines)

		space_lines_with_lot_image = cv2.addWeighted(parking_lot_image, 0.8, parking_space_lines_image, 1, 1)

		averaged_space_lines = self.average_lines(space_lines)

		parking_space_average_lines_image = self.display_lines(parking_lot_image, averaged_space_lines)

		averaged_space_lines_with_lot_image = cv2.addWeighted(parking_lot_image, 0.8, parking_space_average_lines_image, 1, 1)

		# cluster_polygons is FINAL coordinates of polygon blocks
		# blocks_image, cluster_lines, cluster_polygons = draw_blocks(parking_lot_image, space_lines)
		blocks_image, cluster_lines, cluster_polygons = self.draw_blocks(parking_lot_image, averaged_space_lines)

		# cluster_lines_new is clusters map in form (x1, y1, x2, y2)
		# middle_block_lines is FINAL coordinates of middle of parking blocks
		all_spots_image, middle_block_lines, spot_divider_lines, cluster_lines_new = self.create_parking(blocks_image, cluster_lines, cluster_polygons)

		# final_spot_lines is FINAL parking spot dividers
		final_image, final_spot_lines = self.create_final_parking(parking_lot_image, cluster_polygons, middle_block_lines, spot_divider_lines)

		spots_map, count_map, total_count = self.create_spots_ids(cluster_polygons, middle_block_lines, final_spot_lines)

		final_image_with_ids = self.draw_spot_ids(final_image, spots_map)

		predicted_image = self.predict_on_lot(final_image_with_ids, spots_map)

		# cv2.imshow("Result1", parking_lot_image)
		# cv2.imshow("Result2", canny_edges)
		# cv2.imshow("Result3", cropped_image)
		# cv2.imshow("Result4", parking_space_lines_image)
		# cv2.imshow("Result5", space_lines_with_lot_image)
		# cv2.imshow("Result6", parking_space_average_lines_image)
		# cv2.imshow("Result7", averaged_space_lines_with_lot_image)
		# cv2.imshow("Result8", blocks_image)
		# cv2.imshow("Result9", all_spots_image)
		# cv2.imshow("Result10", final_image)
		# cv2.imshow("Result12", final_image_with_ids)
		cv2.imshow("Result13", predicted_image)
		cv2.waitKey(0)


def main():
	# parking_lot = Parking_Lot()
	onlyfiles = [f for f in listdir("/Users/abhishek.venkatesh/Desktop/Capstone/Data/model_images") if isfile(join("/Users/abhishek.venkatesh/Desktop/Capstone/Data/model_images", f))]
	for file in onlyfiles:
		filename = "/Users/abhishek.venkatesh/Desktop/Capstone/Data/model_images/"+file
		parking_lot = Parking_Lot(filename)
		parking_lot.execute_lots()

if __name__ == "__main__":
	main()
	# pass




