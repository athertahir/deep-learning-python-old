# %%
'''
## Local Centering
The example below calculates the mean for each color channel in the loaded image, then
centers the pixel values for each channel separately. Note that NumPy allows us to specify
the dimensions over which a statistic like the mean, min, and max are calculated via the axis
argument. In this example, we set this to (0,1) for the width and height dimensions, which
leaves the third dimension or channels. The result is one mean, min, or max for each of the
three channel arrays.
Also note that when we calculate the mean that we specify the dtype as ‘float64’; this is
required as it will cause all sub-operations of the mean, such as the sum, to be performed with
64-bit precision. Without this, the sum will be performed at lower resolution and the resulting
mean will be wrong given the accumulated errors in the loss of precision, in turn meaning the
mean of the centered pixel values for each channel will not be zero (or a very small number
close to zero).
'''

# %%
# example of per-channel centering (subtract mean)
from numpy import asarray
from PIL import Image
# load image
image = Image.open('sydney_bridge.jpg')
pixels = asarray(image)
# convert from integers to floats
pixels = pixels.astype('float32')
# calculate per-channel means and standard deviations
means = pixels.mean(axis=(0,1), dtype='float64')
print('Means: %s' % means)
print('Mins: %s, Maxs: %s' % (pixels.min(axis=(0,1)), pixels.max(axis=(0,1))))
# per-channel centering of pixels
pixels -= means
# confirm it had the desired effect
means = pixels.mean(axis=(0,1), dtype='float64')
print('Means: %s' % means)
print('Mins: %s, Maxs: %s' % (pixels.min(axis=(0,1)), pixels.max(axis=(0,1))))

# %%
'''
Running the example first reports the mean pixels values for each channel, as well as the
min and max values for each channel. The pixel values are centered, then the new means and
min/max pixel values across each channel are reported. We can see that the new mean pixel
values are very small numbers close to zero and the values are negative and positive values
centered on zero.
'''
