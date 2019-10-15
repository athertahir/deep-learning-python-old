# %%
'''
## Local Standardization
The example below calculates the mean and standard deviation of the loaded image per-channel,
then uses these statistics to standardize the pixels separately in each channel.
'''

# %%
'''
Running the example first calculates and reports the means and standard deviation of
the pixel values in each channel. The pixel values are then standardized and statistics are
re-calculated, confirming the new zero-mean and unit standard deviation.
'''

# %%
# example of per-channel pixel standardization
from numpy import asarray
from PIL import Image
# load image
image = Image.open('sydney_bridge.jpg')
pixels = asarray(image)
# convert from integers to floats
pixels = pixels.astype('float32')
# calculate per-channel means and standard deviations
means = pixels.mean(axis=(0,1), dtype='float64')
stds = pixels.std(axis=(0,1), dtype='float64')
print('Means: %s, Stds: %s' % (means, stds))
# per-channel standardization of pixels
pixels = (pixels - means) / stds
# confirm it had the desired effect
means = pixels.mean(axis=(0,1), dtype='float64')
stds = pixels.std(axis=(0,1), dtype='float64')
print('Means: %s, Stds: %s' % (means, stds))