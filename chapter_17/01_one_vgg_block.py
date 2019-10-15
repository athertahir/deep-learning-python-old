# %%
'''
## How to implement VGG Blocks
The VGG convolutional neural network architecture, named for the Visual Geometry Group at
Oxford, was an important milestone in the use of deep learning methods for computer vision
(introduced in Chapter 15). The architecture was described in the 2014 paper titled Very Deep
Convolutional Networks for Large-Scale Image Recognition by Karen Simonyan and Andrew
Zisserman and achieved top results in the LSVRC-2014 computer vision competition. The key
innovation in this architecture was the definition and repetition of what we will refer to as
VGG-blocks. These are groups of convolutional layers that use small filters (e.g. 3 x 3 pixels)
followed by a max pooling layer.

The image is passed through a stack of convolutional (conv.) layers, where we use
filters with a very small receptive field: 3 x 3 (which is the smallest size to capture
the notion of left/right, up/down, center). [...] Max-pooling is performed over a
2 x 2 pixel window, with stride 2.

— Very Deep Convolutional Networks for Large-Scale Image Recognition, 2014.

A convolutional neural network with VGG-blocks is a sensible starting point when developing
a new model from scratch as it is easy to understand, easy to implement, and very effective at
extracting features from images. We can generalize the specification of a VGG-block as one or
more convolutional layers with the same number of filters and a filter size of 3 x 3, a stride of
1 x 1, same padding so the output size is the same as the input size for each filter, and the use
of a rectified linear activation function. These layers are then followed by a max pooling layer
with a size of 2 x 2 and a stride of the same dimensions. We can define a function to create a
VGG-block using the Keras functional API with a given number of convolutional layers and
with a given number of filters per layer.
'''

# %%
'''
To use the vgg_block function, one would pass in the layer prior to the block and receive the layer
for the end of the block that can be used to integrate into the model. For example, the first
layer might be an input layer which could be passed into the function as an argument. The
function then returns a reference to the final layer in the block, the pooling layer, that could be
connected to a flatten layer and subsequent dense layers for making a classification prediction.
We can demonstrate how to use this function by defining a small model that expects square
color images as input and adds a single VGG block to the model with two convolutional layers,
each with 64 filters
'''

# %%
# Example of creating a CNN model with a VGG block
from keras.models import Model
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.utils import plot_model

# function for creating a vgg block
def vgg_block(layer_in, n_filters, n_conv):
	# add convolutional layers
	for _ in range(n_conv):
		layer_in = Conv2D(n_filters, (3,3), padding='same', activation='relu')(layer_in)
	# add max pooling layer
	layer_in = MaxPooling2D((2,2), strides=(2,2))(layer_in)
	return layer_in

# define model input
visible = Input(shape=(256, 256, 3))
# add vgg module
layer = vgg_block(visible, 64, 2)
# create model
model = Model(inputs=visible, outputs=layer)
# summarize model
model.summary()
# plot model architecture
plot_model(model, show_shapes=True, to_file='/tmp/vgg_block.png')

# %%
'''
Running the example creates the model and summarizes the structure. We can see that,
as intended, the model added a single VGG block with two convolutional layers each with 64
filters, followed by a max pooling layer.
'''
