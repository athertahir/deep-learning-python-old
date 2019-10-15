# %%
'''
## Fix the Border Effect Problem With Padding
By default, a filter starts at the left of the image with the left-hand side of the filter sitting on
the far left pixels of the image. The filter is then stepped across the image one column at a
time until the right-hand side of the filter is sitting on the far right pixels of the image. An
alternative approach to applying a filter to an image is to ensure that each pixel in the image is
given an opportunity to be at the center of the filter. By default, this is not the case, as the
pixels on the edge of the input are only ever exposed to the edge of the filter. By starting the
filter outside the frame of the image, it gives the pixels on the border of the image more of an
opportunity for interacting with the filter, more of an opportunity for features to be detected by
the filter, and in turn, an output feature map that has the same shape as the input image.

For example, in the case of applying a 3 X 3 filter to the 8 X 8 input image, we can add a
border of one pixel around the outside of the image. This has the effect of artificially creating
a 10 X 10 input image. When the 3 X 3 filter is applied, it results in an 8 X 8 feature map.
The added pixel values could have the value zero value that has no effect with the dot product
operation when the filter is applied.
'''

# %%
'''
The addition of pixels to the edge of the image is called padding. In Keras, this is specified
via the padding argument on the Conv2D layer, which has the default value of ‘valid’ (no
padding). This means that the filter is applied only to valid ways to the input. The padding
value of ‘same’ calculates and adds the padding required to the input image (or feature map)
to ensure that the output has the same shape as the input. The example below adds padding to
the convolutional layer in our worked example.
'''

# %%
# %%
# example a convolutional layer with padding
from keras.models import Sequential
from keras.layers import Conv2D
# create model
model = Sequential()
model.add(Conv2D(1, (3,3), padding='same', input_shape=(8, 8, 1)))
# summarize model
model.summary()