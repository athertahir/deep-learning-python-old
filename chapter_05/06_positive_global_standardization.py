# %%
'''
## Positive Global Standardization
There may be a desire to maintain the pixel values in the positive domain, perhaps so the images
can be visualized or perhaps for the benefit of a chosen activation function in the model. A
popular way of achieving this is to clip the standardized pixel values to the range [-1, 1] and then
rescale the values from [-1,1] to [0,1]. The example below updates the global standardization
example to demonstrate this additional rescaling.
'''

# %%
'''
Running the example first calculates the global mean and standard deviation pixel values,
standardizes the pixel values, then confirms the transform by reporting the new global mean
and standard deviation of 0.0 and 1.0 respectively.
'''

# %%
# example of global pixel standardization shifted to positive domain
from numpy import asarray
from numpy import clip
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
# clip pixel values to [-1,1]
pixels = clip(pixels, -1.0, 1.0)
# shift from [-1,1] to [0,1] with 0.5 mean
pixels = (pixels + 1.0) / 2.0
# confirm it had the desired effect
mean, std = pixels.mean(), pixels.std()
print('Mean: %.3f, Standard Deviation: %.3f' % (mean, std))
print('Min: %.3f, Max: %.3f' % (pixels.min(), pixels.max()))
