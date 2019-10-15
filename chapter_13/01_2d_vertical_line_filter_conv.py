# %%
'''
## Detecting Vertical Lines
Before we look at some examples of pooling layers and their effects, let’s develop a small example
of an input image and convolutional layer to which we can later add and evaluate pooling layers. In this example, we define a single input
image or sample that has one channel and is an 8 pixel by 8 pixel square with all 0 values and a
two-pixel wide vertical line in the center
'''

# %%
# example of vertical line detection with a convolutional layer
from numpy import asarray
from keras.models import Sequential
from keras.layers import Conv2D
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


# %%
'''
Next, we can define a model that expects input samples to have the shape (8, 8, 1) and has
a single hidden convolutional layer with a single filter with the shape of 3 pixels by 3 pixels.
A rectified linear activation function, or ReLU for short, is then applied to each value in the
feature map. This is a simple and effective nonlinearity, that in this case will not change the
values in the feature map, but is present because we will later add subsequent pooling layers
and pooling is added after the nonlinearity applied to the feature maps, e.g. a best practice.
'''

# %%
# create model
model = Sequential()
model.add(Conv2D(1, (3,3), activation='relu', input_shape=(8, 8, 1)))
# summarize model
model.summary()

# %%
'''
The filter is initialized with random weights as part of the initialization of the model. Instead,
we will hard code our own 3 X 3 filter that will detect vertical lines. That is, the filter will
strongly activate when it detects a vertical line and weakly activate when it does not. We expect
that by applying this filter across the input image that the output feature map will show that
the vertical line was detected.

Next, we can apply the filter to our input image by calling the predict() function on the
model.
'''

# %%
# define a vertical line detector
detector = [[[[0]],[[1]],[[0]]],
            [[[0]],[[1]],[[0]]],
            [[[0]],[[1]],[[0]]]]
weights = [asarray(detector), asarray([0.0])]
# store the weights in the model
model.set_weights(weights)
# apply filter to input data
yhat = model.predict(data)


# %%
'''
The result is a four-dimensional output with one batch, a given number of rows and columns,
and one filter, or [batch, rows, columns, filters]. We can print the activations in the single
feature map to confirm that the line was detected.
'''

# %%
# enumerate rows
for r in range(yhat.shape[1]):
	# print each column in the row
	print([yhat[0,r,c,0] for c in range(yhat.shape[2])])

# %%
'''
Running the example first summarizes the structure of the model. Of note is that the single
hidden convolutional layer will take the 8 X 8 pixel input image and will produce a feature map
with the dimensions of 6 X 6. We can also see that the layer has 10 parameters: that is nine
weights for the filter (3 X 3) and one weight for the bias. Finally, the single feature map is
printed. We can see from reviewing the numbers in the 6 X 6 matrix that indeed the manually
specified filter detected the vertical line in the middle of our input image
'''