# %%
'''
OpenCV provides the CascadeClassifier class that can be used to create a cascade
classifier for face detection. The constructor can take a filename as an argument that specifies
the XML file for a pre-trained model. OpenCV provides a number of pre-trained models as
part of the installation. These are available on your system and are also available on the
OpenCV GitHub project. We can load the model as follows:

# load the pre-trained model
classifier = CascadeClassifier('haarcascade_frontalface_default.xml')

Once loaded, the model can be used to perform face detection on a photograph by calling
the detectMultiScale() function. This function will return a list of bounding boxes for all
faces detected in the photograph.
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
pixels = imread('test2.jpg')
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


# %%
'''
Running the example, we can see that many of the faces were detected correctly, but the
result is not perfect. We can see that a face on the first or bottom row of people was detected
twice, that a face on the middle row of people was not detected, and that the background on
the third or top row was detected as a face.
'''