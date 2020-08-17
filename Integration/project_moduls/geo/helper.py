import matplotlib.pyplot as plt
import numpy as np
import cv2
import operator
import math

import statistics 
from statistics import mode
from statistics import mean

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

import pickle

dist1 = lambda line1, line2: math.sqrt(  (line2.mid_x - line1.mid_x)**2 + (line2.mid_y - line1.mid_y)**2  )


class Line():
	lineCount = 0

	def __init__(self, x1, y1, x2, y2):
		self.rho = 5
		self.x1 = x1
		self.x2 = x2
		self.y1 = y1
		self.y2 = y2
		Line.lineCount += 1
		self.idx = Line.lineCount

	@property
	def mid_x(self):
		return int((self.x1+self.x2)/2)

	@property
	def mid_y(self):
		return int((self.y1+self.y2)/2)

	@property
	def length(self):
		return int( math.sqrt( (self.y2-self.y1)**2 + (self.x2-self.x1)**2 ) )

	@property
	def ori(self):
		return int( math.degrees(math.atan2(abs((self.x2 - self.x1)), abs((self.y2-self.y1)))) )
		# return int( math.degrees(math.atan2((self.x2 - self.x1), (self.y2-self.y1))) )
	@property
	def ori2(self):
		dx = self.x2 - self.x1
		dy = self.y2 - self.y1
		sign1 = dx*dy
		o = int( math.degrees(math.atan2(abs(dx), abs(dy))) )
		o = o if sign1<=0 else 180-o
		self.sign = -1 if sign1>0 else 1
		o = o if o<=90 else 180-o
		return o

	# @property
	# def dfo(self):
	# 	return int( math.sqrt(  (self.mid_y)**2  +  (self.mid_x)**2   ) )

	# @property
	# def ori_o(self):
	# 	return int( math.degrees(math.atan2(self.mid_x, self.mid_y)) )

	# @property
	# def r(self):
	# 	return int(self.dfo/self.rho) if self.dfo%self.rho <5 else int(self.dfo/self.rho)+1

	# @property
	# def v(self):
	# 	return self.dfo + self.ori_o

	# @property
	# def v1(self):
	# 	return self.r*self.rho + self.ori_o

	# @property
	# def v2(self):
	# 	return self.r*self.rho + math.sqrt(2*(self.r**2)*(1-math.cos(self.ori_o)))

		# Line.lineCount += 1


	# def dist(self, center):
	# 	return int( math.sqrt(  (center[0]-self.mid_x)**2 + (center[1]-self.mid_y)**2  ))
 
	# def print_line(self):
	# 	print(f'id = {self.idx}')
	# 	print(f'point1 = ({self.x1},{self.y1})')
	# 	print(f'point2 = ({self.x2},{self.y2})')  
	# 	print(f'center = ({self.mid_x},{self.mid_y})')
	# 	print(f'length = {self.length}')
	# 	print(f'orientation = {self.ori}')
	# 	print(f'orientation_origin = {self.ori_o}')
	# 	print(f'distance from origin = {self.dfo}')
	# 	print(f'virtual distance = {self.v}')
	# 	print(f'virtual distance 1 = {self.v1}')
	# 	print(f'virtual distance 2 = {self.v2}\n')

def show_images(image1, title1, image2, title2):
 	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
 	ax1.imshow(image1)
 	ax1.axis("off")
 	ax1.set_title(title1)
	#, cmap='gray'
 	ax2.imshow(image2)
 	ax2.axis("off")
 	ax2.set_title(title2)
 	plt.tight_layout()
 	plt.show()

def show_one(image1, title1):
	#,cmap = 'gray'
	plt.imshow(image1)
	plt.axis("off")
	plt.title(title1)
	plt.tight_layout()
	plt.show()

def show(image1, title1, image2, title2, image3, title3, image4, title4):
	fig,a =  plt.subplots(2,2)

	a[0][0].imshow(image1)
	a[0][0].set_title(title1)
	a[0][0].axis("off")

	a[0][1].imshow(image2)
	a[0][1].set_title(title2)
	a[0][1].axis("off")

	a[1][0].imshow(image3)
	a[1][0].set_title(title3)
	a[1][0].axis("off")

	a[1][1].imshow(image4)
	a[1][1].set_title(title4)
	a[1][1].axis("off")

	# plt.tight_layout()
	plt.show()

def roi(image):
	mask = np.ones_like(image)
	masked_image = cv2.bitwise_and(image, mask)
	return masked_image

def my_canny(image):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(gray,(5,5),0)	
	# canny = cv2.Canny(blur, 50, 150)
	canny = cv2.Canny(blur, 75, 150)
	return canny

def process_lines(lines):
	new = []
	# i = 1
	if lines is not None:
		for line in lines:
			for x1,y1,x2,y2 in line:
				# new.append(Line(x1,y1,x2,y2,i))
				new.append(Line(x1,y1,x2,y2))
				# i+=1
	return new

def display_lines(image, lines, color=[255, 0, 0], thickness=2):
	# the lines returned by cv2.HoughLinesP has the shape (-1, 1, 4)
	line_image = np.zeros_like(image)

	for line in lines:
		if 45 < line.ori < 135:
			cv2.line(line_image, (line.x1, line.y1), (line.x2, line.y2), (0,0,255), thickness)
		else:
			cv2.line(line_image, (line.x1, line.y1), (line.x2, line.y2), color, thickness)
	
		cv2.circle(line_image, (line.mid_x,line.mid_y), 1, (0,255,0), 2)
		cv2.circle(line_image, (line.mid_x,line.mid_y), 10, (0,255,0), 1)

		# cv2.circle(line_image, (line.x1, line.y1), 1, (0,255,0), 3) # start point
		# cv2.circle(line_image, (line.x2, line.y2), 1, (0,0,255), 3) # end point

		# cv2.putText(line_image,str(line.ori),(line.x2-10,line.y2-10),cv2.FONT_HERSHEY_SIMPLEX,0.3,(255,255,0),1,cv2.LINE_AA)

		# pts = np.array([[line.mid_x-10,line.mid_y-20],[line.mid_x-20,line.mid_y+10],[line.mid_x-10,line.mid_y+20],[line.mid_x+20,line.mid_y-10]], np.int32)
		# uncomment
		# pts = np.array([[line.x2,line.y2],[line.x1,line.y1],[line.x1+20*math.cos(math.radians(line.ori)),line.y1+20*math.sin(math.radians(line.ori))],[line.x2+20*math.cos(math.radians(line.ori)),line.y2+20*math.sin(math.radians(line.ori))]], np.int32)
		# pts = pts.reshape((-1,1,2))
		# cv2.polylines(line_image,[pts],True,(0,255,255))
	
	return line_image

def display_parking_spots(length, ori, image, lines, color=[255, 0, 0], thickness=2):
	# the lines returned by cv2.HoughLinesP has the shape (-1, 1, 4)
	line_image = np.zeros_like(image)
	c,h,w = image.shape
	for line in lines:
		if 45 < line.ori < 135:
			cv2.line(line_image, (line.x1, line.y1), (line.x2, line.y2), (0,0,255), thickness)
		else:
			cv2.line(line_image, (line.x1, line.y1), (line.x2, line.y2), color, thickness)
	
		cv2.circle(line_image, (line.mid_x,line.mid_y), 1, (0,255,0), 2)
		cv2.circle(line_image, (line.mid_x,line.mid_y), 10, (0,255,0), 1)

		# cv2.circle(line_image, (line.x1, line.y1), 1, (0,255,0), 3) # start point
		# cv2.circle(line_image, (line.x2, line.y2), 1, (0,0,255), 3) # end point
		# cv2.putText(line_image,str(line.ori2),(line.x2-10,line.y2-10),cv2.FONT_HERSHEY_SIMPLEX,0.3,(255,255,0),1,cv2.LINE_AA)
		
		extra_length = 2
		extra_ori = 0
		extra_width = 10
		# print(line.ori2)
		# print(math.radians(line.ori2))
		# print(math.cos(math.radians(line.ori2)))
		# print(math.sin(math.radians(line.ori2)))
		# print('\n')
		# o = line.ori2
		o= 90-line.sign*ori
		# print(line.sign)
		x1 = line.mid_x + (extra_length+length/2)*math.cos(math.radians(o+extra_ori))
		y1 = line.mid_y - (extra_length+length/2)*math.sin(math.radians(o+extra_ori))
		x2 = line.mid_x - (extra_length+length/2)*math.cos(math.radians(o+extra_ori))
		y2 = line.mid_y + (extra_length+length/2)*math.sin(math.radians(o+extra_ori))
		x3 = x2 + (extra_width+extra_length+length/2)*math.sin(math.radians(o+extra_ori))
		y3 = y2 + (extra_width+extra_length+length/2)*math.cos(math.radians(o+extra_ori))
		x4 = x1 + (extra_width+extra_length+length/2)*math.sin(math.radians(o+extra_ori))
		y4 = y1 + (extra_width+extra_length+length/2)*math.cos(math.radians(o+extra_ori))

		# displays the corners of the bounding box
		# cv2.circle(line_image, (int(x1),int(y1)), 1, (204,0,104), 2)
		# cv2.circle(line_image, (int(x2),int(y2)), 1, (255,255,0), 2)
		# cv2.circle(line_image, (int(x3),int(y3)), 1, (51,153,255), 2)
		# cv2.circle(line_image, (int(x4),int(y4)), 1, (0,153,0), 2)

		pts = np.array([[x1,y1],[x2,y2],[x3,y3],[x4,y4]],np.int32)
		pts = pts.reshape((-1,1,2))
		cv2.polylines(line_image,[pts],True,(0,255,255))

		x1_2 = line.mid_x + (extra_length+length/2)*math.cos(math.radians(o+extra_ori))
		y1_2 = line.mid_y - (extra_length+length/2)*math.sin(math.radians(o+extra_ori))
		x2_2 = line.mid_x - (extra_length+length/2)*math.cos(math.radians(o+extra_ori))
		y2_2 = line.mid_y + (extra_length+length/2)*math.sin(math.radians(o+extra_ori))
		x3_2 = x2_2 - (extra_width+extra_length+length/2)*math.sin(math.radians(o+extra_ori))
		y3_2 = y2_2 - (extra_width+extra_length+length/2)*math.cos(math.radians(o+extra_ori))
		x4_2 = x1_2 - (extra_width+extra_length+length/2)*math.sin(math.radians(o+extra_ori))
		y4_2 = y1_2 - (extra_width+extra_length+length/2)*math.cos(math.radians(o+extra_ori))


		pts_2 = np.array([[x1_2,y1_2],[x2_2,y2_2],[x3_2,y3_2],[x4_2,y4_2]],np.int32)
		pts_2 = pts_2.reshape((-1,1,2))
		cv2.polylines(line_image,[pts_2],True,(0,255,255))
	
	return line_image

def display_parking_spots1(length, ori, image, lines, color=[255, 0, 0], thickness=2):
	# the lines returned by cv2.HoughLinesP has the shape (-1, 1, 4)
	line_image = np.zeros_like(image)
	# spot_dict = {} # maps each parking ID to its coords
	# print(f'ori = {ori}\n')
	coords = []
	# with open("coords.txt","w+") as file:
	for line in lines:
		if 45 < line.ori < 135:
			cv2.line(line_image, (line.x1, line.y1), (line.x2, line.y2), (0,0,255), thickness)
		else:
			cv2.line(line_image, (line.x1, line.y1), (line.x2, line.y2), color, thickness)
	
		cv2.circle(line_image, (line.mid_x,line.mid_y), 1, (0,255,0), 2)
		cv2.circle(line_image, (line.mid_x,line.mid_y), 10, (0,255,0), 1)

		# cv2.circle(line_image, (line.x1, line.y1), 1, (0,255,0), 3) # start point
		# cv2.circle(line_image, (line.x2, line.y2), 1, (0,0,255), 3) # end point
		# cv2.putText(line_image,str(line.ori2),(line.x2-10,line.y2-10),cv2.FONT_HERSHEY_SIMPLEX,0.3,(255,255,0),1,cv2.LINE_AA)
		# print(line.ori2)

		extra_length = 2
		extra_ori = 0
		# o = line.ori2
		# o = 90 - line.ori2
		o= 90-line.sign*ori
		# print(line.sign)
		x1 = line.mid_x + (extra_length+length/2)*math.cos(math.radians(o+extra_ori))
		y1 = line.mid_y - (extra_length+length/2)*math.sin(math.radians(o+extra_ori))
		x2 = line.mid_x - (extra_length+length/2)*math.cos(math.radians(o+extra_ori))
		y2 = line.mid_y + (extra_length+length/2)*math.sin(math.radians(o+extra_ori))
		x3 = x2 + (extra_length+length/2)*math.sin(math.radians(o+extra_ori))
		y3 = y2 + (extra_length+length/2)*math.cos(math.radians(o+extra_ori))
		x4 = x1 + (extra_length+length/2)*math.sin(math.radians(o+extra_ori))
		y4 = y1 + (extra_length+length/2)*math.cos(math.radians(o+extra_ori))

		# displays the corners of the bounding box
		# cv2.circle(line_image, (int(x1),int(y1)), 1, (204,0,104), 2)
		# cv2.circle(line_image, (int(x2),int(y2)), 1, (255,255,0), 2)
		# cv2.circle(line_image, (int(x3),int(y3)), 1, (51,153,255), 2)
		# cv2.circle(line_image, (int(x4),int(y4)), 1, (0,153,0), 2)

		pts = np.array([[x1,y1],[x2,y2],[x3,y3],[x4,y4]],np.int32)
		pts = pts.reshape((-1,1,2))
		cv2.polylines(line_image,[pts],True,(0,255,255))

		# x2_2 = line.mid_x + (extra_length+length/2)*math.sin(math.radians(ori+extra_ori))
		# y2_2 = line.mid_y - (extra_length+length/2)*math.cos(math.radians(ori+extra_ori))
		# x1_2 = line.mid_x - (extra_length+length/2)*math.sin(math.radians(ori+extra_ori))
		# y1_2 = line.mid_y + (extra_length+length/2)*math.cos(math.radians(ori+extra_ori))
		# x3_2 = x1_2 - (extra_length+length/2)*math.cos(math.radians(ori+extra_ori))
		# y3_2 = y1_2 - (extra_length+length/2)*math.sin(math.radians(ori+extra_ori))
		# x4_2 = x2_2 - (extra_length+length/2)*math.cos(math.radians(ori+extra_ori))
		# y4_2 = y2_2 - (extra_length+length/2)*math.sin(math.radians(ori+extra_ori))

		# pts_2 = np.array([[x2_2,y2_2],[x1_2,y1_2],[x3_2,y3_2],[x4_2,y4_2]],np.int32)
		# pts_2 = pts_2.reshape((-1,1,2))
		# cv2.polylines(line_image,[pts_2],True,(0,255,255))

		coords.append( [  (int(x1),int(y1)),(int(x2),int(y2)),(int(x3),int(y3)),(int(x4),int(y4))  ] )
		# spot_dict[(int(x2),int(y2),int(x1),int(y1),int(x3),int(y3),int(x4),int(y4))] = line.idx
		# file.write(f'Box {line.idx}: ({int(x2)},{int(y2)}), ({int(x1)},{int(y1)}), ({int(x3)},{int(y3)}), ({int(x4)},{int(y4)})\n')
	
	# with open("spots.pickle","wb") as pickle_out:
	# 	pickle.dump(spot_dict, pickle_out, protocol=pickle.HIGHEST_PROTOCOL)
	# 	pickle_out.close()

	return coords, line_image

# def mode_filtering(lines, criteria:str, DOF):
# 	values = [line.ori if criteria== 'ORI' else line.length for line in lines]
# 	m = mode(values)
# 	if criteria == 'ORI':
# 		filtered = list(filter(lambda line:  m-DOF <= line.ori <= m+DOF , lines))
# 	else:
# 		filtered = list(filter(lambda line:  int(m-m*DOF) <= line.length <= int(m+m*DOF) , lines))
	
# 	return filtered, m

def clus(lines):
	small = []
	big = []
	ignore = []
	for i in range(len(lines)):
		if lines[i].idx not in ignore:
			small.append(lines[i])
			for j in range(i+1, len(lines)):
				if dist1(lines[i], lines[j]) < 20 and lines[j].idx not in ignore:
					ignore.append(lines[j].idx)
					small.append(lines[j])
			big.append(small)
		small = []


	reduced = []
	for group in big:
		# for each group of lines take the line with maximum length
		reduced.append(max(group, key=lambda line: line.length))

	return reduced

# def closest_point(point, points):
# 	points = np.asarray(points)
# 	dist_2 = np.sum((points - point)**2, axis=1)
# 	return np.argmin(dist_2)

# def find_center(lines):
# 	points = np.array([(x.mid_x,x.mid_y) for x in lines ])
# 	# points = np.asarray(points)
# 	center = np.mean(points, axis = 0)
# 	return (int(center[0]),int(center[1]))

class Box():

	def __init__(self, p1, p2, p3, p4):
		self.lines_idx = []
		self.p1 = p1
		self.p2 = p2
		self.p3 = p3
		self.p4 = p4

	# @property
	def contain(self, line):
		point = Point(line.mid_x, line.mid_y)
		polygon = Polygon([(self.p1[0], self.p1[1]), (self.p2[0], self.p2[1]), (self.p3[0], self.p3[1]), (self.p4[0], self.p4[1])])
		if polygon.contains(point):
			self.lines_idx.append(line)

def find_contours(image, lines, th:int):
	boxes = []
	imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	ret, thresh = cv2.threshold(imgray, 127, 255, 0)
	# output, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	# output, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	

	#output, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	# Nutov --> removed element
	contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	# output, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
	# c = max(contours, key = cv2.contourArea)
	cnts = sorted(contours, key = cv2.contourArea, reverse = True)

	# for cnt in cnts:
	# 	rect = cv2.minAreaRect(cnt)
	# 	box = cv2.boxPoints(rect)
	# 	box = np.int0(box)
	# 	print(box[0])
	# 	cv2.drawContours(image,[box],0,(0,0,255),2)


	for b in cnts:
		rect = cv2.minAreaRect(b)
		# print(rect)
		box = cv2.boxPoints(rect)
		box = np.int0(box)
		# print(box[0],box[1],box[2],box[3])
		cv2.drawContours(image,[box],0,(0,0,255),2)
		boxes.append(Box(box[0],box[1],box[2],box[3]))
	
	# show_one(image, 'with clustering')

	for line in lines:
		for bb in boxes:
			bb.contain(line)
			# dist = cv2.pointPolygonTest(bb,(line.mid_x,line.mid_y),True)
			# dist = cv2.pointPolygonTest(cnt,(50,50),True)


	final = []
	for b in boxes:
		size = len(b.lines_idx)
		if  size > th:
			for line in b.lines_idx:
				final.append(line)
		else:
			break

	return final

def hull(image):
	pass
	#Convex Hull