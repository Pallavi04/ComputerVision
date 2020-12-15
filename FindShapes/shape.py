import numpy as np
import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i","--image",required = True, help="Path of Image File")
args = vars(ap.parse_args())

#image = cv2.imread("image.png")
print("Path: ", args["image"])
image = cv2.imread(args["image"])

# find all the 'black' shapes in the image
upper = np.array([15,15,15])
lower = np.array([0,0,0])
shapeMask = cv2.inRange(image,lower,upper)

# find the contours in the mask
cnts = cv2.findContours(shapeMask.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
print("Found {} black shapes".format(len(cnts)))
cv2.imshow("Mask", shapeMask)
# loop over the contours
for c in cnts:
	# draw the contour and show it
	cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
	cv2.imshow("Image", image)
	cv2.waitKey(0)