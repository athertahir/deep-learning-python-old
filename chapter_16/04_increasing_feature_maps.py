# %%
'''
## Example of Increasing Feature Maps
The 1X 1 filter can be used to increase the number of feature maps. This is a common operation
used after a pooling layer prior to applying another convolutional layer. The projection effect of
the filter can be applied as many times as needed to the input, allowing the number of feature
maps to be scaled up and yet have a composition that captures the salient features of the
original. We can increase the number of feature maps from 512 input from the first hidden layer
to double the size at 1,024 feature maps.

model.add(Conv2D(1024, (1,1), activation='relu'))

The complete example is listed below
'''

# %%
# example of a 1x1 filter to increase dimensionality
from keras.models import Sequential
from keras.layers import Conv2D
# create model
model = Sequential()
model.add(Conv2D(512, (3,3), padding='same', activation='relu', input_shape=(256, 256, 3)))
model.add(Conv2D(1024, (1,1), activation='relu'))
# summarize model
model.summary()

# %%
'''
Running the example creates the model and summarizes its structure. We can see that the
width and height of the feature maps are unchanged and that the number of feature maps was
increased from 512 to double the size at 1,024.
'''