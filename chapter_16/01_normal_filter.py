# %%
'''
## Examples of How to Use 1x1 Convolutions
We can make the use of a 1 X 1 filter concrete with some examples. Consider that we have
a convolutional neural network that expected color images input with the square shape of
256 X 256 X 3 pixels. These images then pass through a first hidden layer with 512 filters, each
with the size of 3 X 3 with the same padding, followed by a ReLU activation function. The
example below demonstrates this simple model
'''

# %%
'''
Running the example creates the model and summarizes the model architecture. There
are no surprises; the output of the first hidden layer is a block of feature maps with the
three-dimensional shape of 256 X 256 X 512.
'''

# %%
# example of simple cnn model
from keras.models import Sequential
from keras.layers import Conv2D
# create model
model = Sequential()
model.add(Conv2D(512, (3,3), padding='same', activation='relu', input_shape=(256, 256, 3)))
# summarize model
model.summary()