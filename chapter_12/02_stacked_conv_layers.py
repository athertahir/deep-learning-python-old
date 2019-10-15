# %%
'''
## Problem of Border Effects
In the previous section, we defined a single filter with the size of three pixels high and three
pixels wide (rows, columns). We saw that the application of the 3 X 3 filter, referred to as the
kernel size in Keras, to the 8 X 8 input image resulted in a feature map with the size of 6 X 6.
That is, the input image with 64 pixels was reduced to a feature map with 36 pixels. Where did
the other 28 pixels go?

The filter is applied systematically to the input image. It starts at the top left corner of the
image and is moved from left to right one pixel column at a time until the edge of the filter
reaches the edge of the image. For a 3 X 3 pixel filter applied to a 8 X 8 input image, we can
see that it can only be applied six times, resulting in the width of six in the output feature map.
'''

# %%
'''
The reduction in the size of the input to the feature map is referred to as border effects.
It is caused by the interaction of the filter with the border of the image. This is often not a
problem for large images and small filters but can be a problem with small images. It can also
become a problem once a number of convolutional layers are stacked. For example, below is the
same model updated to have two stacked convolutional layers. This means that a 3 X 3 filter is
applied to the 8 X 8 input image to result in a 6 X 6 feature map as in the previous section. A
3 X 3 filter is then applied to the 6 X 6 feature map
'''

# %%
# example of stacked convolutional layers
from keras.models import Sequential
from keras.layers import Conv2D
# create model
model = Sequential()
model.add(Conv2D(1, (3,3), input_shape=(8, 8, 1)))
model.add(Conv2D(1, (3,3)))
# summarize model
model.summary()


# %%
'''
Running the example summarizes the shape of the output from each layer. We can see that
the application of filters to the feature map output of the first layer, in turn, results in a 4 X 4
feature map. This can become a problem as we develop very deep convolutional neural network
models with tens or hundreds of layers. We will simply run out of data in our feature maps
upon which to operate.
'''