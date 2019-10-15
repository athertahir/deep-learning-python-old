# %%
'''
## OpenCV
It is a modestly complex classifier that has also been tweaked and refined over the last
nearly 20 years. A modern implementation of the Classifier Cascade face detection algorithm is
provided in the OpenCV library. This is a C++ computer vision library that provides a Python
interface. The benefit of this implementation is that it provides pre-trained face detection
models, and provides an interface to train a model on your own dataset.
'''

# %%
# example of face detection with opencv cascade classifier
from cv2 import imread
from cv2 import CascadeClassifier
# load the photograph
pixels = imread('test1.jpg')


# %%
'''
OpenCV provides the CascadeClassifier class that can be used to create a cascade
classifier for face detection. The constructor can take a filename as an argument that specifies
the XML file for a pre-trained model. OpenCV provides a number of pre-trained models as
part of the installation. These are available on your system and are also available on the
OpenCV GitHub project. We can load the model as follows:

classifier = CascadeClassifier('haarcascade_frontalface_default.xml')

Once loaded, the model can be used to perform face detection on a photograph by calling
the detectMultiScale() function. This function will return a list of bounding boxes for all
faces detected in the photograph.
'''


# %%
# load the pre-trained model
classifier = CascadeClassifier('haarcascade_frontalface_default.xml')
# perform face detection
bboxes = classifier.detectMultiScale(pixels)
# print bounding box for each detected face
for box in bboxes:
	print(box)


# %%
'''
The example of performing face detection on the college students photograph with
a pre-trained cascade classifier in OpenCV is listed above.

Running the example first loads the photograph, then loads and configures the cascade
classifier; faces are detected and each bounding box is printed. Each box lists the x and y
coordinates for the bottom-left-hand-corner of the bounding box, as well as the width and the
height. The results suggest that two bounding boxes were detected.
'''