# %%
'''
## How to Implement the Inception Module
The inception module was described and used in the GoogLeNet model in the 2015 paper by
Christian Szegedy, et al. titled Going Deeper with Convolutions (introduced in Chapter 15).
Like the VGG model, the GoogLeNet model achieved top results in the 2014 version of the
ILSVRC challenge. The key innovation on the inception model is called the inception module.
This is a block of parallel convolutional layers with different sized filters (e.g. 1 x 1, 3 x 3, 5 x 5)
and a 3 x 3 max pooling layer, the results of which are then concatenated.

This is a very simple and powerful architectural unit that allows the model to learn not only
parallel filters of the same size, but parallel filters of differing sizes, allowing learning at multiple
scales. We can implement an inception module directly using the Keras functional API. The
function below will create a single inception module with a specified number of filters for each
of the parallel convolutional layers. From the GoogLeNet architecture described in the paper, it
does not appear to use a systematic number of filters for parallel convolutional layers as the
model is highly optimized. As such, we can parameterize the module definition so that we can
specify the number of filters to use in each of the 1 x 1, 3 x 3, and 5 x 5 filters.
'''

# %%
'''
If you intend to use many inception modules in your model, you may require this computational performance-based modification. The function below implements this optimization
improvement with parameterization so that you can control the amount of reduction in the
number of filters prior to the 3 x 3 and 5 x 5 convolutional layers and the number of increased
filters after max pooling.


We can create a model with two of these optimized inception modules to get a concrete idea
of how the architecture looks in practice.
'''

# %%
# example of creating a CNN with an efficient inception module
from keras.models import Model
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers.merge import concatenate
from keras.utils import plot_model

# function for creating a projected inception module
def inception_module(layer_in, f1, f2_in, f2_out, f3_in, f3_out, f4_out):
	# 1x1 conv
	conv1 = Conv2D(f1, (1,1), padding='same', activation='relu')(layer_in)
	# 3x3 conv
	conv3 = Conv2D(f2_in, (1,1), padding='same', activation='relu')(layer_in)
	conv3 = Conv2D(f2_out, (3,3), padding='same', activation='relu')(conv3)
	# 5x5 conv
	conv5 = Conv2D(f3_in, (1,1), padding='same', activation='relu')(layer_in)
	conv5 = Conv2D(f3_out, (5,5), padding='same', activation='relu')(conv5)
	# 3x3 max pooling
	pool = MaxPooling2D((3,3), strides=(1,1), padding='same')(layer_in)
	pool = Conv2D(f4_out, (1,1), padding='same', activation='relu')(pool)
	# concatenate filters, assumes filters/channels last
	layer_out = concatenate([conv1, conv3, conv5, pool], axis=-1)
	return layer_out

# define model input
visible = Input(shape=(256, 256, 3))
# add inception block 1
layer = inception_module(visible, 64, 96, 128, 16, 32, 32)
# add inception block 1
layer = inception_module(layer, 128, 128, 192, 32, 96, 64)
# create model
model = Model(inputs=visible, outputs=layer)
# summarize model
model.summary()
# plot model architecture
plot_model(model, show_shapes=True, to_file='/tmp/inception_module.png')

# %%
'''
Running the example creates a linear summary of the layers that does not really help to
understand what is going on. The output is omitted here for brevity. A plot of the model
architecture is created that does make the layout of each module clear and how the first model
feeds the second module. Note that the first 1 x 1 convolution in each inception module is on
the far right for space reasons, but besides that, the other layers are organized left to right
within each module
'''