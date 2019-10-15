# %%
'''
## Effect of Filter Size (Kernel Size)
Different sized filters will detect different sized features in the input image and, in turn, will
result in differently sized feature maps. It is common to use 3 X 3 sized filters, and perhaps
5 X 5 or even 7 X 7 sized filters, for larger input images. For example, below is an example of
the model with a single filter updated to use a filter size of 8 X 8 pixels.
'''

# %%
# example of a convolutional layer
from keras.models import Sequential
from keras.layers import Conv2D
# create model
model = Sequential()
model.add(Conv2D(1, (8,8), input_shape=(8, 8, 1)))
# summarize model
model.summary()

# %%
'''
Running the example, we can see that, as you might expect, there is one filter weight for
each pixel in the input image (64 + 1 for the bias) and that the output is a feature map with a
single pixel
'''