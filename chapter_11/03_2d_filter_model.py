# %%
'''
## Example of 2D Convolutional Layer
We can expand the bump detection example in the previous section to a vertical line detector in
a two-dimensional image. Again, we can constrain the input, in this case to a square 8 X 8 pixel
input image with a single channel (e.g. grayscale) with a single vertical line in the middle.


[0, 0, 0, 1, 1, 0, 0, 0]

[0, 0, 0, 1, 1, 0, 0, 0]

[0, 0, 0, 1, 1, 0, 0, 0]

[0, 0, 0, 1, 1, 0, 0, 0]

[0, 0, 0, 1, 1, 0, 0, 0]

[0, 0, 0, 1, 1, 0, 0, 0]

[0, 0, 0, 1, 1, 0, 0, 0]

[0, 0, 0, 1, 1, 0, 0, 0]


The input to a Conv2D layer must be four-dimensional. The first dimension defines the
samples; in this case, there is only a single sample. The second dimension defines the number of
rows; in this case, eight. The third dimension defines the number of columns, again eight in this
case, and finally the number of channels, which is one in this case. Therefore, the input must
have the four-dimensional shape [samples, columns, rows, channels] or [1, 8, 8, 1] in
this case.
'''

# %%
'''
We will define the Conv2D layer with a single filter as we did in the previous section with
the Conv1D example. The filter will be two-dimensional and square with the shape 3 X 3. The
layer will expect input samples to have the shape [columns, rows, channels] or [8,8,1].
'''

# %%
# example of calculation 2d convolutions
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
We will define a vertical line detector filter to detect the single vertical line in our input
data. The filter looks as follows:

0, 1, 0

0, 1, 0

0, 1, 0

Finally, we will apply the filter to the input image, which will result in a feature map that
we would expect to show the detection of the vertical line in the input image.

The shape of the feature map output will be four-dimensional with the shape [batch, rows,
columns, filters]. We will be performing a single batch and we have a single filter (one filter
and one input channel), therefore the output or feature map shape is [1, 6, 6, 1]. We can
pretty-print the content of the single feature map as follows:
'''


# %%
# create model
model = Sequential()
model.add(Conv2D(1, (3,3), input_shape=(8, 8, 1)))
# define a vertical line detector
detector = [[[[0]],[[1]],[[0]]],
            [[[0]],[[1]],[[0]]],
            [[[0]],[[1]],[[0]]]]
weights = [asarray(detector), asarray([0.0])]
# store the weights in the model
model.set_weights(weights)
# confirm they were stored
print(model.get_weights())
# apply filter to input data
yhat = model.predict(data)


# %%
'''
The shape of the feature map output will be four-dimensional with the shape [batch, rows,
columns, filters]. We will be performing a single batch and we have a single filter (one filter
and one input channel), therefore the output or feature map shape is [1, 6, 6, 1]. We can
pretty-print the content of the single feature map as follows:
'''

# %%
for r in range(yhat.shape[1]):
	# print each column in the row
	print([yhat[0,r,c,0] for c in range(yhat.shape[2])])

# %%
'''
Running the example first confirms that the handcrafted filter was correctly defined in the
layer weights. Next, the calculated feature map is printed. We can see from the scale of the
numbers that indeed the filter has detected the single vertical line with strong activation in the
middle of the feature map.
'''