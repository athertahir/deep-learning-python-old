# %%
'''
## Center Pixel Values
A popular data preparation technique for image data is to subtract the mean value from the
pixel values. This approach is called centering, as the distribution of the pixel values is centered
on the value of zero. Centering can be performed before or after normalization. Centering the
pixels then normalizing will mean that the pixel values will be centered close to 0.5 and be in
the range 0-1. Centering after normalization will mean that the pixels will have positive and
negative values, in which case images will not display correctly (e.g. pixels are expected to have
value in the range 0-255 or 0-1). Centering after normalization might be preferred, although it
might be worth testing both approaches.
'''

# %%
'''

Centering requires that a mean pixel value be calculated prior to subtracting it from the
pixel values. There are multiple ways that the mean can be calculated; for example:

- Per image.
- Per minibatch of images (under stochastic gradient descent).5.4. Center Pixel Values 49
- Per training dataset.

The mean can be calculated for all pixels in the image, referred to as a global centering, or
it can be calculated for each channel in the case of color images, referred to as local centering.

- Global Centering: Calculating and subtracting the mean pixel value across color
channels.
- Local Centering: Calculating and subtracting the mean pixel value per color channel.

Per-image global centering is common because it is trivial to implement. Also common is
per minibatch global or local centering for the same reason: it is fast and easy to implement.
In some cases, per-channel means are pre-calculated across an entire training dataset. In this
case, the image means must be stored and used both during training and any inference with
the trained models in the future. For models trained on images centered using these means
that may be used for transfer learning on new tasks, it can be beneficial or even required to
normalize images for the new task using the same means. Let’s look at a few examples.
'''

# %%
'''
## Global Centering
The example below calculates a global mean across all three color channels in the loaded image,
then centers the pixel values using the global mean.
'''

# %%
# example of global centering (subtract mean)
from numpy import asarray
from PIL import Image
# load image
image = Image.open('sydney_bridge.jpg')
pixels = asarray(image)
# convert from integers to floats
pixels = pixels.astype('float32')
# calculate global mean
mean = pixels.mean()
print('Mean: %.3f' % mean)
print('Min: %.3f, Max: %.3f' % (pixels.min(), pixels.max()))
# global centering of pixels
pixels = pixels - mean
# confirm it had the desired effect
mean = pixels.mean()
print('Mean: %.3f' % mean)
print('Min: %.3f, Max: %.3f' % (pixels.min(), pixels.max()))

# %%
'''
## Global Centering
Running the example, we can see that the mean pixel value is about 152. Once centered,
we can confirm that the new mean for the pixel values is 0.0 and that the new data range is
negative and positive around this mean.
'''
