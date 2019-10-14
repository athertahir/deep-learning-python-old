# %%
'''
We can update the face detector example to plot the photograph and draw each bounding box. This can
be achieved by drawing a rectangle for each box directly over the pixels of the loaded image
using the rectangle() function that takes two points.
'''


# %%
'''
Running the example, we can see that the photograph was plotted correctly and that each
face was correctly detected.
'''

# %%
# plot photo with detected faces using opencv cascade classifier
from cv2 import imread
from cv2 import imshow
from cv2 import waitKey
from cv2 import destroyAllWindows
from cv2 import CascadeClassifier
from cv2 import rectangle
# load the photograph
pixels = imread('test1.jpg')
# load the pre-trained model
classifier = CascadeClassifier('haarcascade_frontalface_default.xml')
# perform face detection
bboxes = classifier.detectMultiScale(pixels)
# print bounding box for each detected face
for box in bboxes:
	# extract
	x, y, width, height = box
	x2, y2 = x + width, y + height
	# draw a rectangle over the pixels
	rectangle(pixels, (x, y), (x2, y2), (0,0,255), 1)
# show the image
imshow('face detection', pixels)
# keep the window open until we press a key
# waitKey(0)
# # close the window
# destroyAllWindows()