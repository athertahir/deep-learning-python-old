# %%
'''
## Random Rotation Augmentation
A rotation augmentation randomly rotates the image clockwise by a given number of degrees
from 0 to 360. The rotation will likely rotate pixels out of the image frame and leave areas of
the frame with no pixel data that must be filled in. The example below demonstrates random
rotations via the rotation range argument, with rotations to the image between 0 and 90
degrees
'''

# %%
'''
Running the example generates examples of the rotated image, showing in some cases pixels
rotated out of the frame and the nearest-neighbor fill
'''

# %%
# example of random rotation image augmentation
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
datagen = ImageDataGenerator(rotation_range=90)
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