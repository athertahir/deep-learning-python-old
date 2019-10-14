# %%
'''
## Effect of Filter Size (Kernel Size)
Different sized filters will detect different sized features in the input image and, in turn, will
result in differently sized feature maps. It is common to use 3 × 3 sized filters, and perhaps
5 × 5 or even 7 × 7 sized filters, for larger input images. For example, below is an example of
the model with a single filter updated to use a filter size of 5 × 5 pixels.
'''

# %%
# example of a convolutional layer
from keras.models import Sequential
from keras.layers import Conv2D
# create model
model = Sequential()
model.add(Conv2D(1, (5,5), input_shape=(8, 8, 1)))
# summarize model
model.summary()

# %%
'''
Running the example demonstrates that the 5 × 5 filter can only be applied to the 8 × 8
input image 4 times, resulting in a 4 × 4 feature map output.
'''