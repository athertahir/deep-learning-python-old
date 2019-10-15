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
We can now try face detection on the swim team photograph, e.g. the image test2.jpg.
Running the example, we can see that all thirteen faces were correctly detected and that it
looks roughly like all of the facial keypoints are also correct.

Running the example creates a plot that shows each separate face detected in the photograph
of the swim team.
'''

# %%
# extract and plot each detected face in a photograph
%matplotlib notebook
from matplotlib import pyplot
from mtcnn.mtcnn import MTCNN

# draw each face separately
def draw_faces(filename, result_list):
	# load the image
	data = pyplot.imread(filename)
	# plot each face as a subplot
	for i in range(len(result_list)):
		# get coordinates
		x1, y1, width, height = result_list[i]['box']
		x2, y2 = x1 + width, y1 + height
		# define subplot
		pyplot.subplot(1, len(result_list), i+1)
		pyplot.axis('off')
		# plot face
		pyplot.imshow(data[y1:y2, x1:x2])
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
draw_faces(filename, faces)