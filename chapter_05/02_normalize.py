# %%
'''
## Normalize Pixel Values
For most image data, the pixel values are integers with values between 0 and 255. Neural
networks process inputs using small weight values, and inputs with large integer values can
disrupt or slow down the learning process. As such it is good practice to normalize the pixel
values so that each pixel value has a value between 0 and 1. It is valid for images to have pixel
values in the range 0-1 and images can be viewed normally.

This can be achieved by dividing all pixels values by the largest pixel value; that is 255.
This is performed across all channels, regardless of the actual range of pixel values that are
present in the image. The example below loads the image and converts it into a NumPy array.
The data type of the array is reported and the minimum and maximum pixel values across all
three channels are then printed. Next, the array is converted to the float data type before the
pixel values are normalized and the new range of pixel values is reported.
'''

# %%
'''
Running the example prints the data type of the NumPy array of pixel values, which we can
see is an 8-bit unsigned integer.
The min and maximum pixel values are printed, showing the expected 0 and 255 respectively.
The pixel values are normalized and the new minimum and maximum of 0.0 and 1.0 are then
reported.
'''

# %%
# example of pixel normalization
from numpy import asarray
from PIL import Image
# load image
image = Image.open('sydney_bridge.jpg')
pixels = asarray(image)
# confirm pixel range is 0-255
print('Data Type: %s' % pixels.dtype)
print('Min: %.3f, Max: %.3f' % (pixels.min(), pixels.max()))
# convert from integers to floats
pixels = pixels.astype('float32')
# normalize to the range 0-1
pixels /= 255.0
# confirm the normalization
print('Min: %.3f, Max: %.3f' % (pixels.min(), pixels.max()))


# %%
'''
Normalization is a good default data preparation that can be performed if you are in doubt
as to the type of data preparation to perform. It can be performed per image and does not
require the calculation of statistics across the training dataset, as the range of pixel values is a
domain standard.
'''