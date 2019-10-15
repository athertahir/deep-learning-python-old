# %%
'''
## Downsample Input With Stride
The filter is moved across the image left to right, top to bottom, with a one-pixel column change
on the horizontal movements, then a one-pixel row change on the vertical movements. The12.6. Downsample Input With Stride 130
amount of movement between applications of the filter to the input image is referred to as the
stride, and it is almost always symmetrical in height and width dimensions. The default stride
or strides in two dimensions is (1,1) for the height and the width movement, performed when
needed. And this default works well in most cases. The stride can be changed, which has an
effect both on how the filter is applied to the image and, in turn, the size of the resulting feature
map.

For example, the stride can be changed to (2,2). This has the effect of moving the filter
two pixels left for each horizontal movement of the filter and two pixels down for each vertical
movement of the filter when creating the feature map. We can demonstrate this with an example
using the 8 X 8 image with a vertical line (left) dot product (. operator) with the vertical line
filter (right) with a stride of two pixels.
'''

# %%
'''
We can see that there are only three valid applications of the 3 X 3 filters to the 8 X 8 input
image with a stride of two. This will be the same in the vertical dimension. This has the effect
of applying the filter in such a way that the normal feature map output (6 X 6) is down-sampled
so that the size of each dimension is reduced by half (3 X 3), resulting in 1 4 the number of
pixels (36 pixels down to 9). The stride can be specified in Keras on the Conv2D layer via the
stride argument and specified as a tuple with height and width. The example demonstrates
the application of our manual vertical line filter on the 8 X 8 input image with a convolutional
layer that has a stride of two.
'''

# %%
# example of vertical line filter with a stride of 2
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
# create model
model = Sequential()
model.add(Conv2D(1, (3,3), strides=(2, 2), input_shape=(8, 8, 1)))
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
# enumerate rows
for r in range(yhat.shape[1]):
	# print each column in the row
	print([yhat[0,r,c,0] for c in range(yhat.shape[2])])

# %%
'''
Running the example, we can see from the summary of the model that the shape of the
output feature map will be 3 X 3. Applying the handcrafted filter to the input image and
printing the resulting activation feature map, we can see that, indeed, the filter still detected
the vertical line, and can represent this finding with less information. Downsampling may be
desirable in some cases where deeper knowledge of the filters used in the model or of the model
architecture allows for some compression in the resulting feature maps.
'''
