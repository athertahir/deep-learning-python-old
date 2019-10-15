# %%
'''
## Average Pooling Layer
On two-dimensional feature maps, pooling is typically applied in 2 X 2 patches of the feature
map with a stride of (2,2). Average pooling involves calculating the average for each patch of
the feature map. This means that each 2 X 2 square of the feature map is downsampled to the
average value in the square. For example, the output of the line detector convolutional filter13.4. Average Pooling Layer 138
in the previous section was a 6 X 6 feature map. We can look at applying the average pooling
operation to the first line of patches of that feature map manually. The first line of pooling
input (first two rows and six columns) of the output feature map were as follows:

[0.0, 0.0, 3.0, 3.0, 0.0, 0.0]
[0.0, 0.0, 3.0, 3.0, 0.0, 0.0]

'''

# %%
# example of average pooling
from numpy import asarray
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import AveragePooling2D
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
model.add(AveragePooling2D())
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
Running the example first summarizes the model. We can see from the model summary that
the input to the pooling layer will be a single feature map with the shape (6,6) and that the
output of the average pooling layer will be a single feature map with each dimension halved,
with the shape (3,3). Applying the average pooling results in a new feature map that still
detects the line, although in a downsampled manner, exactly as we expected from calculating
the operation manually
'''