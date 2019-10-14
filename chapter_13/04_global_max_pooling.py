# %%
'''
## Global Pooling Layers
There is another type of pooling that is sometimes used called global pooling. Instead of down
sampling patches of the input feature map, global pooling downsamples the entire feature map to
a single value. This would be the same as setting the pool size to the size of the input feature
map. Global pooling can be used in a model to aggressively summarize the presence of a feature
in an image. It is also sometimes used in models as an alternative to using a fully connected
layer to transition from feature maps to an output prediction for the model. Both global average
pooling and global max pooling are supported by Keras via the GlobalAveragePooling2D and
GlobalMaxPooling2D classes respectively. For example, we can add global max pooling to the
convolutional model used for vertical line detection.
'''

# %%
'''
The outcome will be a single value that will summarize the strongest activation or presence
of the vertical line in the input image. The complete code listing is provided below.
'''

# %%
# example of using global max pooling
from numpy import asarray
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import GlobalMaxPooling2D
# define input data
data = [[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0]]
data = asarray(data)
data = data.reshape(1, 8, 8, 1)
# create model
model = Sequential()
model.add(Conv2D(1, (3,3), activation='relu', input_shape=(8, 8, 1)))
model.add(GlobalMaxPooling2D())
# summarize model
model.summary()
# define a vertical line detector
detector = [[[[0]],[[1]],[[0]]],
            [[[0]],[[1]],[[0]]],
            [[[0]],[[1]],[[0]]]]
weights = [asarray(detector), asarray([0.0])]
# store the weights in the model
model.set_weights(weights)
# apply filter to input data
yhat = model.predict(data)
# show result
print(yhat)


# %%
'''
Running the example first summarizes the model. We can see that, as expected, the output
of the global pooling layer is a single value that summarizes the presence of the feature in the
single feature map. Next, the output of the model is printed showing the effect of global max
pooling on the feature map, printing the single largest activation
'''