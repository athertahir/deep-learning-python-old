# %%
'''
## Effect of Filter Size (Kernel Size)
Different sized filters will detect different sized features in the input image and, in turn, will
result in differently sized feature maps. It is common to use 3 X 3 sized filters, and perhaps
5 X 5 or even 7 X 7 sized filters, for larger input images. For example, below is an example of
the model with a single filter updated to use a filter size of 1 X 1 pixels.
'''

# %%
# example of a convolutional layer
from keras.models import Sequential
from keras.layers import Conv2D
# create model
model = Sequential()
model.add(Conv2D(1, (1,1), input_shape=(8, 8, 1)))
# summarize model
model.summary()

# %%
'''
Running the example demonstrates that the output feature map has the same size as the
input, specifically 8 X 8. This is because the filter only has a single weight (and a bias)
'''