# %%
'''
## Face Detection With Deep Learning
A number of deep learning methods have been developed and demonstrated for face detection.
Perhaps one of the more popular approaches is called the Multi-Task Cascaded Convolutional
Neural Network, or MTCNN for short, described by Kaipeng Zhang, et al. in the 2016 paper
titled Joint Face Detection and Alignment Using Multitask Cascaded Convolutional Networks.
The MTCNN is popular because it achieved then state-of-the-art results on a range of benchmark
datasets, and because it is capable of also recognizing other facial features such as eyes and
mouth, called landmark detection.
The network uses a cascade structure with three networks; first the image is rescaled to a
range of different sizes (called an image pyramid), then the first model (Proposal Network or
P-Net) proposes candidate facial regions, the second model (Refine Network or R-Net) filters the
bounding boxes, and the third model (Output Network or O-Net) proposes facial landmarks.
'''

# %%
'''
The example plots the photograph again with bounding boxes and facial key points. We
can see that eyes, nose, and mouth are detected well on each face, although the mouth on the
right face could be better detected, with the points looking a little lower than the corners of the
mouth.
'''

# %%
# face detection with mtcnn on a photograph
%matplotlib notebook
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from mtcnn.mtcnn import MTCNN

# draw an image with detected objects
def draw_image_with_boxes(filename, result_list):
	# load the image
	data = pyplot.imread(filename)
	# plot the image
	pyplot.imshow(data)
	# get the context for drawing boxes
	ax = pyplot.gca()
	# plot each box
	for result in result_list:
		# get coordinates
		x, y, width, height = result['box']
		# create the shape
		rect = Rectangle((x, y), width, height, fill=False, color='red')
		# draw the box
		ax.add_patch(rect)
		# draw the dots
		for _, value in result['keypoints'].items():
			# create and draw dot
			dot = Circle(value, radius=2, color='red')
			ax.add_patch(dot)
	# show the plot
	pyplot.show()

filename = 'test2.jpg'
# load image from file
pixels = pyplot.imread(filename)
# create the detector, using default weights
detector = MTCNN()
# detect faces in the image
faces = detector.detect_faces(pixels)
# display faces on the original image
draw_image_with_boxes(filename, faces)