# %%
'''
## Example of Decreasing Feature Maps
The 1 X 1 filter can be used to decrease the number of feature maps. This is the most common
application of this type of filter and in this way, the layer is often called a feature map pooling
layer. In this example, we can decrease the depth (or channels) from 512 to 64. This might be
useful if the subsequent layer we were going to add to our model would be another convolutional
layer with 7 X 7 filters. These filters would only be applied at a depth of 64 rather than 512.

model.add(Conv2D(64, (1,1), activation='relu'))


The composition of the 64 feature maps is not the same as the original 512, but contains a
useful summary of dimensionality reduction that captures the salient features, such that the 7X7
operation may have a similar effect on the 64 feature maps as it might have on the original 512.
Further, a 7 X 7 convolutional layer with 64 filters itself applied to the 512 feature maps output
by the first hidden layer would result in approximately one million parameters (activations). If
the 1 X 1 filter is used to reduce the number of feature maps to 64 first, then the number of
parameters required for the 7 X 7 layer is only approximately 200,000, an enormous difference.
The complete example of using a 1 X 1 filter for dimensionality reduction is listed below.
'''

# %%
# example of a 1x1 filter for dimensionality reduction
from keras.models import Sequential
from keras.layers import Conv2D
# create model
model = Sequential()
model.add(Conv2D(512, (3,3), padding='same', activation='relu', input_shape=(256, 256, 3)))
model.add(Conv2D(64, (1,1), activation='relu'))
# summarize model
model.summary()

# %%
'''
Running the example creates the model and summarizes its structure. We can see that
the width and height of the feature maps are unchanged, yet the number of feature maps was
reduced from 512 to 64.
'''