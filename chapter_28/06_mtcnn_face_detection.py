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
We can use mtcnn for face detection. An instance of the network can be created by calling the MTCNN() constructor. By
default, the library will use the pre-trained model, although you can specify your own model
via the weights file argument and specify a path or URL, for example:

model = MTCNN(weights_file='filename.npy')


The minimum box size for detecting a face can be specified via the min face size argument,
which defaults to 20 pixels. The constructor also provides a scale factor argument to specify
the scale factor for the input image, which defaults to 0.709. Once the model is configured and
loaded, it can be used directly to detect faces in photographs by calling the detect faces()
function. This returns a list of dict objects, each providing a number of keys for the details of
each face detected, including:

- 'box': Providing the x, y of the bottom left of the bounding box, as well as the width
and height of the box.
- 'confidence': The probability confidence of the prediction.
- 'keypoints': Providing a dict with dots for the 'left eye', 'right eye', 'nose',
'mouth left', and 'mouth right'.


For example, we can perform face detection on the college students photograph as follows:
'''

# %%
# face detection with mtcnn on a photograph
%matplotlib notebook
from matplotlib import pyplot
from mtcnn.mtcnn import MTCNN
# load image from file
filename = 'test1.jpg'
pixels = pyplot.imread(filename)
# create the detector, using default weights
detector = MTCNN()
# detect faces in the image
faces = detector.detect_faces(pixels)
for face in faces:
	print(face)


# %%
'''
Running the example loads the photograph, loads the model, performs face detection, and
prints a list of each face detected.
'''