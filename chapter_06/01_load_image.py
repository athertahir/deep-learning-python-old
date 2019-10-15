# %%
'''
## How to Load an Image with Keras
Keras provides the load img() function for loading an image from file as a PIL image object.
The example below loads the Bondi Beach photograph from file as a PIL image and reports
details about the loaded image.
'''

# %%
'''
Running the example loads the image and reports details about the loaded image. We can
confirm that the image was loaded as a PIL image in JPEG format with RGB channels and the
size of 640 by 427 pixels.
'''

# %%
'''
The load_img() function provides additional arguments that may be useful when loading
the image, such as ‘grayscale’ that allows the image to be loaded in grayscale (defaults to
False), color mode that allows the image mode or channel format to be specified (defaults to
rgb), and target size that allows a tuple of (height, width) to be specified, resizing the image
automatically after being loaded.
'''

# %%
# example of loading an image with the Keras API
from keras.preprocessing.image import load_img
# load the image
img = load_img('bondi_beach.jpg')
# report details about the image
print(type(img))
print(img.format)
print(img.mode)
print(img.size)
# show the image
img.show()

from IPython.display import display # to display images
display(img)