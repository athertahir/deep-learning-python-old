# %%
'''
## Image Augmentation With ImageDataGenerator
The Keras deep learning library provides the ability to use data augmentation automatically
when training a model. This is achieved by using the ImageDataGenerator class. First, the
class must be instantiated and the configuration for the types of data augmentation are specified
by arguments to the class constructor. A range of techniques are supported, as well as pixel
scaling methods. 

We will focus on five main types of data augmentation techniques for image data; specifically:
- Image shifts via the width shift range and height shift range arguments.
- Image flips via the horizontal flip and vertical flip arguments.
- Image rotations via the rotation range argument
- Image brightness via the brightness range argument.
- Image zoom via the zoom range argument.
'''

# %%
'''
## Horizontal and Vertical Shift Augmentation
A shift to an image means moving all pixels of the image in one direction, such as horizontally
or vertically, while keeping the image dimensions the same. This means that some of the
pixels will be clipped off the image and there will be a region of the image where new pixel
values will have to be specified. The width shift range and height shift range arguments
to the ImageDataGenerator constructor control the amount of horizontal and vertical shift
respectively. These arguments can specify a floating point value that indicates the percentage
(between 0 and 1) of the width or height of the image to shift. Alternately, a number of pixels
can be specified to shift the image.

Specifically, a value in the range between no shift and the percentage or number of pixels
will be sampled for each image and the shift performed, e.g. [0, value]. Alternately, you
can specify a tuple or array of the min and max range from which the shift will be shifted; for
example: [-100, 100] or [-0.5, 0.5]. The example below demonstrates a horizontal shift with
the width shift range argument between [-200,200] pixels and generates a plot of generated
images to demonstrate the effect.
'''

# %%
'''
Running the example creates the instance of ImageDataGenerator configured for image
augmentation, then creates the iterator. The iterator is then called nine times in a loop and
each augmented image is plotted. We can see in the plot of the result that a range of different
randomly selected positive and negative horizontal shifts was performed and the pixel values at
the edge of the image are duplicated to fill in the empty part of the image created by the shift.
'''

# %%
# example of horizontal shift image augmentation
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator

%matplotlib notebook
from matplotlib import pyplot
# load the image
img = load_img('bird.jpg')
# convert to numpy array
data = img_to_array(img)
# expand dimension to one sample
samples = expand_dims(data, 0)
# create image data augmentation generator
datagen = ImageDataGenerator(width_shift_range=[-200,200])
# prepare iterator
it = datagen.flow(samples, batch_size=1)
# generate samples and plot
for i in range(9):
	# define subplot
	pyplot.subplot(330 + 1 + i)
	# generate batch of images
	batch = it.next()
	# convert to unsigned integers for viewing
	image = batch[0].astype('uint8')
	# plot raw pixel data
	pyplot.imshow(image)
# show the figure
pyplot.show()

# %%
'''
Running the example creates a plot of images augmented with random positive and negative
vertical shifts. We can see that both horizontal and vertical positive and negative shifts probably
make sense for the chosen photograph, but in some cases, the replicated pixels at the edge of
the image may not make sense to a model.
Note that other fill modes can be specified via fill mode argument.
'''
