# %%
'''
## Sample Image
We need a sample image for testing in this tutorial. We will use a photograph of the Sydney
Harbor Bridge taken by Bernard Spragg. NZ 1 and released under a permissive license.

The example below will load the image, display some properties about the loaded image,
then show the image. This example and the rest of the tutorial assumes that you have the
Pillow Python library installed
'''

# %%
'''
Running the example reports the format of the image, which is JPEG, and the mode, which
is RGB for the three color channels. Next, the size of the image is reported, showing 640 pixels
in width and 374 pixels in height.
'''

# %%
# load and show an image with Pillow
from PIL import Image
from IPython.display import display # to display images

# load the image
image = Image.open('sydney_bridge.jpg')
# summarize some details about the image
print(image.format)
print(image.mode)
print(image.size)
# show the image
# image.show()
display(image)