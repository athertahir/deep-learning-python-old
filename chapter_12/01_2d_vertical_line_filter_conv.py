# %%
'''
## Convolutional Layer
In a convolutional neural network, a convolutional layer is responsible for the systematic
application of one or more filters to an input. The multiplication of the filter to the input
image results in a single output. The input is typically three-dimensional images (e.g. rows,
columns and channels), and in turn, the filters are also three-dimensional with the same number
of channels and fewer rows and columns than the input image. As such, the filter is repeatedly
applied to each part of the input image, resulting in a two-dimensional output map of activations,
called a feature map. Keras provides an implementation of the convolutional layer called a
Conv2D layer.


It requires that you specify the expected shape of the input images in terms of rows (height),
columns (width), and channels (depth) or [rows, columns, channels]. The filter contains
the weights that must be learned during the training of the layer. The filter weights represent
the structure or feature that the filter will detect and the strength of the activation indicates
the degree to which the feature was detected. The layer requires that both the number of filters
be specified and that the shape of the filters be specified. We can demonstrate this with a small
example (intentionally based on the example from Chapter 11). In this example, we define a
single input image or sample that has one channel and is an eight pixel by eight pixel square
with all 0 values and a two-pixel wide vertical line in the center.
'''

# %%
'''
The filter is initialized with random weights as part of the initialization of the model. We
will overwrite the random weights and hard code our own 3 X 3 filter that will detect vertical
lines. That is, the filter will strongly activate when it detects a vertical line and weakly activate
when it does not. We expect that by applying this filter across the input image, the output
feature map will show that the vertical line was detected
'''

# %%
# example of using a single convolutional layer
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
Next, we can define a model that expects input samples to have the shape (8, 8, 1) and
has a single hidden convolutional layer with a single filter with the shape of three pixels by
three pixels.
'''

# %%
# create model
model = Sequential()
model.add(Conv2D(1, (3,3), input_shape=(8, 8, 1)))
# summarize model
model.summary()




# %%
'''
The filter is initialized with random weights as part of the initialization of the model. We
will overwrite the random weights and hard code our own 3 X 3 filter that will detect vertical
lines. That is, the filter will strongly activate when it detects a vertical line and weakly activate
when it does not. We expect that by applying this filter across the input image, the output
feature map will show that the vertical line was detected
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
# enumerate rows
for r in range(yhat.shape[1]):
	# print each column in the row
	print([yhat[0,r,c,0] for c in range(yhat.shape[2])])


# %%
'''
The result is a four-dimensional output with one batch, a given number of rows and columns,
and one filter, or [batch, rows, columns, filters]. We can print the activations in the
single feature map to confirm that the line was detected.
'''


# %%
'''
Running the example first summarizes the structure of the model. Of note is that the single
hidden convolutional layer will take the 8 X 8 pixel input image and will produce a feature map
with the dimensions of 6 X 6. We will go into why this is the case in the next section. We can
also see that the layer has 10 parameters, that is nine weights for the filter (3 X 3) and one
weight for the bias. Finally, the feature map is printed. We can see from reviewing the numbers
in the 6 X 6 matrix that indeed the manually specified filter detected the vertical line in the
middle of our input image.
'''