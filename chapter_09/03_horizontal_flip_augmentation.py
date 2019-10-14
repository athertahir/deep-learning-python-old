# %%
'''
## Horizontal and Vertical Flip Augmentation
An image flip means reversing the rows or columns of pixels in the case of a vertical or
horizontal flip respectively. The flip augmentation is specified by a boolean horizontal flip
or vertical flip argument to the ImageDataGenerator class constructor. For photographs
like the bird photograph used in this tutorial, horizontal flips may make sense, but vertical flips
would not. For other types of images, such as aerial photographs, cosmology photographs, and
microscopic photographs, perhaps vertical flips make sense. The example below demonstrates
augmenting the chosen photograph with horizontal flips via the horizontal flip argument.
'''

# %%
'''
Running the example creates a plot of nine augmented images. We can see that the horizontal
flip is applied randomly to some images and not others
'''

# %%
# example of horizontal flip image augmentation
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
datagen = ImageDataGenerator(horizontal_flip=True)
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