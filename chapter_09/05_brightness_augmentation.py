# %%
'''
## Random Brightness Augmentation
The brightness of the image can be augmented by either randomly darkening images, brightening
images, or both. The intent is to allow a model to generalize across images trained on different
lighting levels. This can be achieved by specifying the brightness range argument to the
ImageDataGenerator() constructor that specifies min and max range as a float representing a
percentage for selecting a darkening or brightening amount. Values less than 1.0 darken the
image, e.g. [0.5, 1.0], whereas values larger than 1.0 brighten the image, e.g. [1.0, 1.5], where 1.0
has no effect on brightness. The example below demonstrates a brightness image augmentation,
allowing the generator to randomly darken the image between 1.0 (no change) and 0.2 or 20%.
'''

# %%
'''
Running the example shows the augmented images with varying amounts of darkening
applied.
'''

# %%
# example of brighting image augmentation
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
datagen = ImageDataGenerator(brightness_range=[0.2,1.0])
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