# %%
'''
## Standardize Pixel Values
The distribution of pixel values often follows a Normal or Gaussian distribution, e.g. bell
shape. This distribution may be present per image, per minibatch of images, or across the
training dataset and globally or per channel. As such, there may be benefit in transforming the
distribution of pixel values to be a standard Gaussian: that is both centering the pixel values on
zero and normalizing the values by the standard deviation. The result is a standard Gaussian
of pixel values with a mean of 0.0 and a standard deviation of 1.0.

As with centering, the operation can be performed per image, per minibatch, and across
the entire training dataset, and it can be performed globally across channels or locally per
channel. Standardization may be preferred to normalization and centering alone and it results
in both zero-centered values and small input values, roughly in the range -3 to 3, depending
on the specifics of the dataset. For consistency of the input data, it may make more sense to
standardize images per-channel using statistics calculated per minibatch or across the training
dataset, if possible. Let’s look at some examples.

'''

# %%
'''
## Global Standardization
The example below calculates the mean and standard deviations across all color channels in the
loaded image, then uses these values to standardize the pixel values
'''

# %%
# example of global pixel standardization
from numpy import asarray
from PIL import Image
# load image
image = Image.open('sydney_bridge.jpg')
pixels = asarray(image)
# convert from integers to floats
pixels = pixels.astype('float32')
# calculate global mean and standard deviation
mean, std = pixels.mean(), pixels.std()
print('Mean: %.3f, Standard Deviation: %.3f' % (mean, std))
# global standardization of pixels
pixels = (pixels - mean) / std
# confirm it had the desired effect
mean, std = pixels.mean(), pixels.std()
print('Mean: %.3f, Standard Deviation: %.3f' % (mean, std))

# %%
'''
Running the example first calculates the global mean and standard deviation pixel values,
standardizes the pixel values, then confirms the transform by reporting the new global mean
and standard deviation of 0.0 and 1.0 respectively.
'''
