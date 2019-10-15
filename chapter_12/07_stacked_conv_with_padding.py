# %%
'''
The addition of padding allows the development of very deep models in such a way that the
feature maps do not dwindle away to nothing. The example below demonstrates this with three
stacked convolutional layers.
'''

# %%
# example a deep cnn with padding
from keras.models import Sequential
from keras.layers import Conv2D
# create model
model = Sequential()
model.add(Conv2D(1, (3,3), padding='same', input_shape=(8, 8, 1)))
model.add(Conv2D(1, (3,3), padding='same'))
model.add(Conv2D(1, (3,3), padding='same'))
# summarize model
model.summary()

# %%
'''
Running the example, we can see that with the addition of padding, the shape of the output
feature maps remains fixed at 8 X 8 even three layers deep.
'''