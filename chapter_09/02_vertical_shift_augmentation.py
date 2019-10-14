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
Below is the same example updated to perform vertical shifts of the image via the height shift range
argument, in this case specifying the percentage of the image to shift as 0.5 the height of the
image.
'''

# %%
# example of vertical shift image augmentation
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
datagen = ImageDataGenerator(height_shift_range=0.5)
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