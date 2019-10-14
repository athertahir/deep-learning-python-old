# %%
'''
## Example of 1D Convolutional Layer
We can define a one-dimensional input that has eight elements; two in the middle with the value
1.0, and three on either side with the value of 0.0.

[0, 0, 0, 1, 1, 0, 0, 0]

The input to Keras must be three dimensional for a 1D convolutional layer. The first
dimension refers to each input sample; in this case, we only have one sample. The second
dimension refers to the length of each sample; in this case, the length is eight. The third
dimension refers to the number of channels in each sample; in this case, we only have a single
channel. Therefore, the shape of the input array will be [1, 8, 1]
'''

# %%
# example of calculation 1d convolutions
from numpy import asarray
from keras.models import Sequential
from keras.layers import Conv1D
# define input data
data = asarray([0, 0, 0, 1, 1, 0, 0, 0])
data = data.reshape(1, 8, 1)


# %%
'''
We will define a model that expects input samples to have the shape [8, 1]. The model will
have a single filter with the shape of 3, or three elements wide. Keras refers to the shape of the
filter as the kernel size (the required second argument to the layer).

Each filter also has a bias input value that also requires a weight that we will set to zero.
Therefore, we can force the weights of our one-dimensional convolutional layer to use our
handcrafted filter as follows:
'''

# %%
# create model
model = Sequential()
model.add(Conv1D(1, 3, input_shape=(8, 1)))
# define a vertical line detector
weights = [asarray([[[0]],[[1]],[[0]]]), asarray([0.0])]
# store the weights in the model
model.set_weights(weights)
# confirm they were stored
print(model.get_weights())


# %%
'''
Finally, we can apply the single filter to our input data. We can achieve this by calling the
predict() function on the model. This will return the feature map directly: that is the output
of applying the filter systematically across the input sequence.
'''

# %%
# apply filter to input data
yhat = model.predict(data)
print(yhat)


# %%
'''
Running the example first prints the weights of the network; that is the confirmation that
our handcrafted filter was set in the model as we expected. Next, the filter is applied to the
input pattern and the feature map is calculated and displayed. We can see from the values of
the feature map that the bump was detected correctly.
'''