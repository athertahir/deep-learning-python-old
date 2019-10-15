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
To use the naive_inception_module function, provide the reference to the prior layer as input, the number of filters,
and it will return a reference to the concatenated filters layer that you can then connect to more
inception modules or a submodel for making a prediction. We can demonstrate how to use this
function by creating a model with a single inception module.
'''

# %%
# example of creating a CNN with an inception module
from keras.models import Model
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers.merge import concatenate
from keras.utils import plot_model

# function for creating a naive inception block
def naive_inception_module(layer_in, f1, f2, f3):
	# 1x1 conv
	conv1 = Conv2D(f1, (1,1), padding='same', activation='relu')(layer_in)
	# 3x3 conv
	conv3 = Conv2D(f2, (3,3), padding='same', activation='relu')(layer_in)
	# 5x5 conv
	conv5 = Conv2D(f3, (5,5), padding='same', activation='relu')(layer_in)
	# 3x3 max pooling
	pool = MaxPooling2D((3,3), strides=(1,1), padding='same')(layer_in)
	# concatenate filters, assumes filters/channels last
	layer_out = concatenate([conv1, conv3, conv5, pool], axis=-1)
	return layer_out

# define model input
visible = Input(shape=(256, 256, 3))
# add inception module
layer = naive_inception_module(visible, 64, 128, 32)
# create model
model = Model(inputs=visible, outputs=layer)
# summarize model
model.summary()
# plot model architecture
plot_model(model, show_shapes=True, to_file='/tmp/naive_inception_module.png')

# %%
'''
Running the example creates the model and summarizes the layers. We know the convolutional and pooling layers are parallel, but this summary does not capture the structure
easily.
'''