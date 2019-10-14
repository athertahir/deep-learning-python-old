# %%
'''
## Random Zoom Augmentation
A zoom augmentation randomly zooms the image and either adds new pixel values around
the image or interpolates pixel values respectively. Image zooming can be configured by the
zoom range argument to the ImageDataGenerator constructor. You can specify the percentage
of the zoom as a single float or a range as an array or tuple. If a float is specified, then the range
for the zoom will be [1-value, 1+value]. For example, if you specify 0.3, then the range will be
[0.7, 1.3], or between 70% (zoom in) and 130% (zoom out). The zoom amount is uniformly
randomly sampled from the zoom region for each dimension (width, height) separately.

The zoom may not feel intuitive. Note that zoom values less than 1.0 will zoom the image
in, e.g. [0.5,0.5] makes the object in the image 50% larger or closer, and values larger than 1.0
will zoom the image out by 50%, e.g. [1.5, 1.5] makes the object in the image smaller or further
away. A zoom of [1.0,1.0] has no effect. The example below demonstrates zooming the image in,
e.g. making the object in the photograph larger.
'''

# %%
# example of zoom image augmentation
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
datagen = ImageDataGenerator(zoom_range=[0.5,1.0])
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
Running the example generates examples of the zoomed image, showing a random zoom
in that is different on both the width and height dimensions that also randomly changes the
aspect ratio of the object in the image.
'''