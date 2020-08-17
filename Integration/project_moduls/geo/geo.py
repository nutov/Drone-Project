import matplotlib.pyplot as plt
import numpy as np
import cv2
import math
import sys
from .helper import *


def GetCoords(src:np.array,is_show = False):
	#------------------------------------------------------------------
	min_length = 20 # 20
	max_length = 200 # 70
	DOF_orientation = 20 # orientation gap in degres
	DOF_length = 0.33 # length gap in percentage
	#------------------------------------------------------------------
	#image_name = sys.argv[1]
	#src = cv2.imread(f'./images/{image_name}')
	


	image = np.copy(src)
	# TODO: delete
	#print(f'image type {image.dtype}')

	canny_image = my_canny(image)
	# TODO: delete
	#print(f'canny_image {canny_image.sum()}')
	#plt.imshow(image)
	#plt.show()
	roi_image = roi(canny_image) 
	# TODO: delete
	#rint(f'roi_image {roi_image.sum()}')
	if is_show:
		show_one(image, 'original')
	#ret,roi_image = cv2.threshold(roi_image,127,255,0)
	#------------------------------------------------------------------
	# morphology
	kernel = np.ones((5,5),np.uint8)
	# roi_image = cv2.morphologyEx(roi_image, cv2.MORPH_GRADIENT, kernel1)
	# kernel2 = np.ones((9,9),np.uint8)
	# roi_image = cv2.morphologyEx(roi_image, cv2.MORPH_TOPHAT, kernel2)
	roi_image = cv2.dilate(roi_image,kernel,iterations = 1)
	roi_image = cv2.erode(roi_image,kernel,iterations = 1)
	if is_show:
		show_one(roi_image, 'roi_image')


	#------------------------------------------------------------------
	# lines = cv2.HoughLinesP(roi_image, 1, np.pi / 180, 20, maxLineGap=10)
	lines = cv2.HoughLinesP(roi_image, 1, np.pi / 180, 20, maxLineGap=10)

	# TODO: delete
	#print(f'lines {lines}')

	all_lines = process_lines(lines)

	# TODO: delete
	#print(f'all_lines {all_lines}')	
	#------------------------------------------------------------------
	# filter out small and large lines
	filtered_lines_by_length = list(filter(lambda line:  min_length <= line.length <= max_length , all_lines))
	# TODO: delete
	#print(f'filtered_lines_by_length {filtered_lines_by_length}')
	#------------------------------------------------------------------
	# mode of orienation
	values = [line.ori for line in filtered_lines_by_length]
	m = mode(values)
	filtered_lines_by_mode_orientation = list(filter(lambda line:  m-DOF_orientation <= line.ori <= m+DOF_orientation , filtered_lines_by_length))
	# filtered_lines_by_mode_orientation, m = mode_filtering(filtered_lines_by_length, 'ORI', DOF_orientation)
	#------------------------------------------------------------------
	# mode of length
	values = [line.length for line in filtered_lines_by_mode_orientation]
	mm = mode(values)
	filtered_lines_by_mode_length = list(filter(lambda line:  int(mm-mm*DOF_length) <= line.length <= int(mm+mm*DOF_length) , filtered_lines_by_mode_orientation))
	# filtered_lines_by_mode_length, mm = mode_filtering(filtered_lines_by_mode_orientation, 'LEN', DOF_length)
	#------------------------------------------------------------------
	### clust = clus(filtered_lines_by_mode_orientation)
	clust = clus(filtered_lines_by_mode_length)
	clustered_lines_by_distance = clus(clust)
	#------------------------------------------------------------------
	vals_length = [line.length for line in clustered_lines_by_distance]
	vals_ori2 = [line.ori2 for line in clustered_lines_by_distance]
	mode_length = mode(vals_length)
	mode_ori2 = mode(vals_ori2)
	# print(f'length = {mode_length} & ori = {mode_ori2}')



	#--------------------------------------
	line_image1 = display_lines(image, all_lines)
	combo_image1 = cv2.addWeighted(image, 0.8, line_image1, 1, 1)

	line_image2 = display_lines(image, filtered_lines_by_length)
	combo_image2 = cv2.addWeighted(image, 0.8, line_image2, 1, 1)

	line_image3 = display_lines(image, filtered_lines_by_mode_orientation)
	combo_image3 = cv2.addWeighted(image, 0.8, line_image3, 1, 1)

	line_image4 = display_lines(image, filtered_lines_by_mode_length)
	combo_image4 = cv2.addWeighted(image, 0.8, line_image4, 1, 1)

	line_image5 = display_lines(image, clustered_lines_by_distance)
	combo_image5 = cv2.addWeighted(image, 0.8, line_image5, 1, 1)

	line_image6 = display_parking_spots(mode_length, mode_ori2, image, clustered_lines_by_distance)
	## combo_image6 = cv2.addWeighted(image, 0.8, line_image6, 1, 1)

	final = find_contours(line_image6, clustered_lines_by_distance, 5)
	vals_ori2 = [line.ori2 for line in final]
	mode_ori3 = mean(vals_ori2)



	coords, line_image7 = display_parking_spots1(mode_length, mode_ori3, image, final)
	combo_image7 = cv2.addWeighted(image, 0.8, line_image7, 1, 1)

	# show(combo_image1, 'all lines',
	#      combo_image2, f'filtered lines by length:\n {min_length}-{max_length}',
	#      combo_image3, f'filtered lines by mode orientation: mode = {m}\n mode-{DOF_orientation} <= orientation <= mode+{DOF_orientation}',
	#      combo_image4, f'filtered lines by mode length: mode = {mm}\n mode-{int(mm*DOF_length)} <= length <= mode+{int(mm*DOF_length)}')



	print('\n')
	print(f'all_lines={len(all_lines)}') 
	print(f'filtered_lines_by_length={len(filtered_lines_by_length)}')
	print(f'filtered_lines_by_mode_orientation={len(filtered_lines_by_mode_orientation)}')
	print(f'filtered_lines_by_mode_length={len(filtered_lines_by_mode_length)}')
	print(f'clustered_lines_by_distance={len(clustered_lines_by_distance)}')
	print(f'final={len(final)}')
	print('\n')
	if is_show:
		show_one(combo_image1, 'all lines')
		show_one(combo_image2, f'filtered lines by length:\n {min_length}-{max_length}')
		show_one(combo_image3, f'filtered lines by mode orientation: mode = {m}\n mode-{DOF_orientation} <= orientation <= mode+{DOF_orientation}')
		show_one(combo_image4, f'filtered lines by mode length: mode = {mm}\n mode-{int(mm*DOF_length)} <= length <= mode+{int(mm*DOF_length)}')

		show_one(combo_image5, 'with clustering')
		show_one(line_image6, 'spots and parking blocks estimation')
		show_one(combo_image7, 'final spots')

	return coords

#if __name__ == "__main__":
#	main()
	