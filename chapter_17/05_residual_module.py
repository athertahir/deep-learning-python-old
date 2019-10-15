# %%
'''
## How to Implement the Residual Module
The Residual Network, or ResNet, architecture for convolutional neural networks was proposed
by Kaiming He, et al. in their 2016 paper titled Deep Residual Learning for Image Recognition,
which achieved success on the 2015 version of the ILSVRC challenge (introduced in Chapter 15).
A key innovation in the ResNet was the residual module. The residual module, specifically the
identity residual model, is a block of two convolutional layers with the same number of filters
and a small filter size where the output of the second layer is added with the input to the first
convolutional layer. Drawn as a graph, the input to the module is added to the output of the
module and is called a shortcut connection. We can implement this directly in Keras using the
functional API and the add() merge function.
'''

# %%
'''
A limitation with this direct implementation is that if the number of filters in the input
layer does not match the number of filters in the last convolutional layer of the module (defined
by n filters), then we will get an error. One solution is to use a 1 x 1 convolution layer,
often referred to as a projection layer, to either increase the number of filters for the input
layer or reduce the number of filters for the last convolutional layer in the module. The former
solution makes more sense, and is the approach proposed in the paper, referred to as a projection
shortcut.

Running the example first creates the model then prints a summary of the layers. Because
the module is linear, this summary is helpful to see what is going on.
'''

# %%
# example of a CNN model with an identity or projection residual module
from keras.models import Model
from keras.layers import Input
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import add
from keras.utils import plot_model

# function for creating an identity or projection residual module
def residual_module(layer_in, n_filters):
	merge_input = layer_in
	# check if the number of filters needs to be increase, assumes channels last format
	if layer_in.shape[-1] != n_filters:
		merge_input = Conv2D(n_filters, (1,1), padding='same', activation='relu', kernel_initializer='he_normal')(layer_in)
	# conv1
	conv1 = Conv2D(n_filters, (3,3), padding='same', activation='relu', kernel_initializer='he_normal')(layer_in)
	# conv2
	conv2 = Conv2D(n_filters, (3,3), padding='same', activation='linear', kernel_initializer='he_normal')(conv1)
	# add filters, assumes filters/channels last
	layer_out = add([conv2, merge_input])
	# activation function
	layer_out = Activation('relu')(layer_out)
	return layer_out

# define model input
visible = Input(shape=(256, 256, 3))
# add vgg module
layer = residual_module(visible, 64)
# create model
model = Model(inputs=visible, outputs=layer)
# summarize model
model.summary()
# plot model architecture
plot_model(model, show_shapes=True, to_file='/tmp/residual_module.png')