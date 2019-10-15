# %%
'''
## Example of Projecting Feature Maps
A 1 X 1 filter can be used to create a projection of the feature maps. The number of feature
maps created will be the same number and the effect may be a refinement of the features already
extracted. This is often called channel-wise pooling, as opposed to traditional feature-wise
pooling on each channel. It can be implemented as follows:

model.add(Conv2D(512, (1,1), activation='relu'))

We can see that we use the same number of features and still follow the application of the
filter with a rectified linear activation function. The complete example is listed below.
'''

# %%
# example of a 1x1 filter for projection
from keras.models import Sequential
from keras.layers import Conv2D
# create model
model = Sequential()
model.add(Conv2D(512, (3,3), padding='same', activation='relu', input_shape=(256, 256, 3)))
model.add(Conv2D(512, (1,1), activation='relu'))
# summarize model
model.summary()

# %%
'''
Running the example creates the model and summarizes the architecture. We can see that
no change is made to the width or height of the feature maps, and by design, the number
of feature maps is kept constant with a simple projection operation applied (e.g. a possible
simplification of the feature maps)
'''