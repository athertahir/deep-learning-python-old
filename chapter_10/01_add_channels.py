# %%
'''
## How to Add a Channel to a Grayscale Image
Grayscale images are loaded as a two-dimensional array. Before they can be used for modeling,
you may have to add an explicit channel dimension to the image. This does not add new
data; instead, it changes the array data structure to have an additional third axis with one
dimension to hold the grayscale pixel values. For example, a grayscale image with the dimensions [rows][cols] can be changed to [rows][cols][channels] or [channels][rows][cols]
where the new [channels] axis has one dimension.
This can be achieved using the expand dims() NumPy function. The axis argument allows
you to specify whether the new dimension will be added to the first, e.g. first for channels
first or last for channels last. The example below loads the Penguin Parade photograph as a
grayscale image using the Pillow library and demonstrates how to add a channel dimension.
'''

# %%
'''
Running the example first loads the photograph using the Pillow library, then converts it to
a grayscale image. The image object is converted to a NumPy array and we confirm the shape
of the array is two dimensional, specifically (424, 640).

The expand dims() function is then used to add a channel via axis=0 to the front of the
array and the change is confirmed with the shape (1, 424, 640). The same function is then
used to add a channel to the end or third dimension of the array with axis=2 and the change is
confirmed with the shape (424, 640, 1).
'''

# %%
# example of expanding dimensions
from numpy import expand_dims
from numpy import asarray
from PIL import Image
# load the image
img = Image.open('penguin_parade.jpg')
# convert the image to grayscale
img = img.convert(mode='L')
# convert to numpy array
data = asarray(img)
print(data.shape)
# add channels first
data_first = expand_dims(data, axis=0)
print(data_first.shape)
# add channels last
data_last = expand_dims(data, axis=2)
print(data_last.shape)