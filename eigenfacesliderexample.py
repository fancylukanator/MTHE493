# Import necessary packages
from __future__ import print_function
import os
import sys
import cv2
import numpy as np

## TRAINING
# Create data matrix from test images
def createDataMatrix(images):

	print("Creating training data matrix",end=" ... ")
	numImages = len(images)
	sz = images[0].shape
	data = np.zeros((numImages, sz[0] * sz[1] * sz[2]), dtype=np.float32)
	for i in range(0, numImages):
		image = images[i].flatten()
		data[i,:] = image
	
	print("DONE")
	return data

# Read training images from the directory
def readImages(path):
	print("Reading training images from " + path, end="...")
	# Create array of array of images.
	images = []
	# List all files in the directory and read points from text files one by one
	for filePath in sorted(os.listdir(path)):
		fileExt = os.path.splitext(filePath)[1]
		if fileExt in [".jpg", ".jpeg", ".png"]:

			# Add to array of images
			imagePath = os.path.join(path, filePath)
			im = cv2.imread(imagePath)

			if im is None :
				print("image:{} not read properly".format(imagePath))
			else :
				# Convert image to floating point
				im = np.float32(im)/255.0
				# Add image to list
				images.append(im)
				# Flip image 
				imFlip = cv2.flip(im, 1);
				# Append flipped image
				images.append(imFlip)
	numImages = int(len(images) / 2)
	# Exit if no image found
	if numImages == 0 :
		print("No images found")
		sys.exit(0)

	print(str(numImages) + " files read.")
	return images

## TESTING
# Read test images from the directory
def readTestImages(path):
	print("Reading test images from " + path, end="...")
	# Create array of array of images.
	testimages = []
	# List all files in the directory and read points from text files one by one
	for filePath in sorted(os.listdir(path)):
		fileExt = os.path.splitext(filePath)[1]
		if fileExt in [".jpg", ".jpeg", ".png"]:

			# Add to array of images
			imagePath = os.path.join(path, filePath)
			im = cv2.imread(imagePath)

			if im is None :
				print("image:{} not read properly".format(imagePath))
			else :
				# Convert image to floating point
				im = np.float32(im)/255.0
				# Add image to list
				testimages.append(im)
				# Flip image 
				imFlip = cv2.flip(im, 1);
				# Append flipped image
				testimages.append(imFlip)
	numTestImages = int(len(testimages) / 2)
	# Exit if no image found
	if numTestImages == 0 :
		print("No images found")
		sys.exit(0)

	print(str(numTestImages) + " files read.")
	return testimages

# Reconstruct Test Face from EigenFaces
#def Reconstruct

## Create face with slider
# Add the weighted eigen faces to the mean face 
def createNewFace(*args):
	# Start with the mean image
	output = averageFace
	
	# Add the eigen faces with the weights
	for i in range(0, NUM_EIGEN_FACES):

		sliderValues[i] = cv2.getTrackbarPos("Weight" + str(i), "Trackbars");
		weight = sliderValues[i] - MAX_SLIDER_VALUE/2
		output = np.add(output, eigenFaces[i] * weight)

	# Display Result at 2x size
	output = cv2.resize(output, (0,0), fx=2, fy=2)
	cv2.imshow("Result", output)

def resetSliderValues(*args):
	for i in range(0, NUM_EIGEN_FACES):
		cv2.setTrackbarPos("Weight" + str(i), "Trackbars", int(MAX_SLIDER_VALUE/2));
	createNewFace()

if __name__ == '__main__':

	# Number of EigenFaces
	NUM_EIGEN_FACES = 10

	# Maximum weight
	MAX_SLIDER_VALUE = 255

	# Directory containing images
	dirName_training = "data\_trainingimages"
	dirName_test = "data\_testimages"

	# Read images
	images = readImages(dirName_training)
	testimages = readTestImages(dirName_test)
	
	# Size of images
	sz = images[0].shape

	# Create data matrix for PCA.
	data = createDataMatrix(images)

	# Compute the eigenvectors from the stack of images created
	print("Calculating PCA ", end="...")
	mean, eigenVectors = cv2.PCACompute(data, mean=None, maxComponents=NUM_EIGEN_FACES)
	print ("DONE")

	averageFace = mean.reshape(sz)

	eigenFaces = [];

	for eigenVector in eigenVectors:
		eigenFace = eigenVector.reshape(sz)
		eigenFaces.append(eigenFace)

	# Create window for displaying Mean Face
	cv2.namedWindow("Mean Face", cv2.WINDOW_AUTOSIZE)
	
	# Display result at 2x size
	output = cv2.resize(averageFace, (0,0), fx=2, fy=2)
	cv2.imshow("Mean Face", output)

	# Create window for displaying result
	cv2.namedWindow("Result", cv2.WINDOW_AUTOSIZE)
	
	# Display result at 2x size
	output = cv2.resize(averageFace, (0,0), fx=2, fy=2)
	cv2.imshow("Result", output)

	# Create Window for trackbars
	cv2.namedWindow("Trackbars", cv2.WINDOW_AUTOSIZE)

	sliderValues = []
	
	# Create Trackbars
	for i in range(0, NUM_EIGEN_FACES):
		sliderValues.append(int(MAX_SLIDER_VALUE/2))
		cv2.createTrackbar( "Weight" + str(i), "Trackbars", int(MAX_SLIDER_VALUE/2), MAX_SLIDER_VALUE, createNewFace)
	
	# You can reset the sliders by clicking on the mean image.
	cv2.setMouseCallback("Mean Face", resetSliderValues);
	
	print('''Usage:
	Change the weights using the sliders
	Click on the result window to reset sliders
	Hit ESC to terminate program.''')
	cv2.waitKey(0)
	cv2.destroyAllWindows()