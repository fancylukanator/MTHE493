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

if __name__ == '__main__':

	# Directory containing images
	dirName_training = "data\_trainingimages"

	# Read images
	images = readImages(dirName_training)

	# Size of images
	sz = images[0].shape

	# Create data matrix for PCA.
	data = createDataMatrix(images)

    # Show Average face
	mean = data.mean(0)
	averageFace = mean.reshape(sz)

    # Create normalized face matrix and covarience matrix
	Norm_Face_Matrix = data - mean
	Norm_Face_Matrix_t = np.transpose(Norm_Face_Matrix)
	CovMatrix = np.matmul(Norm_Face_Matrix, Norm_Face_Matrix_t)

    # get eigen values and eigen vectors
	evals,evects = np.linalg.eig(CovMatrix)
	print("eigenvalues\n",evals)
	print("eigenvectors\n",evects)

    # choose top k eigen vectors based on largest eigen value
    

	# Create window for displaying Mean Face
	cv2.namedWindow("Mean Face", cv2.WINDOW_AUTOSIZE)
	
	# Display result at 2x size
	output = cv2.resize(averageFace, (0,0), fx=2, fy=2)
	cv2.imshow("Mean Face", output)

	cv2.waitKey(0)
	cv2.destroyAllWindows()