# %%
'''
## How to Change Image Channel Ordering
After a color image is loaded as a three-dimensional array, the channel ordering can be changed.
This can be achieved using the moveaxis() NumPy function. It allows you to specify the
index of the source axis and the destination axis. This function can be used to change an
array in channel last format such, as [rows][cols][channels] to channels first format, such
as [channels][rows][cols], or the reverse. The example below loads the Penguin Parade
photograph in channel last format and uses the moveaxis() function change it to channels first
format.
'''

# %%
'''
Running the example first loads the photograph using the Pillow library and converts it to
a NumPy array confirming that the image was loaded in channels last format with the shape
(424, 640, 3). The moveaxis() function is then used to move the channels axis from position
2 to position 0 and the result is confirmed showing channels first format (3, 424, 640). This
is then reversed, moving the channels in position 0 to position 2 again.
'''

# %%
# change image from channels last to channels first format
from numpy import moveaxis
from numpy import asarray
from PIL import Image
# load the color image
img = Image.open('penguin_parade.jpg')
# convert to numpy array
data = asarray(img)
print(data.shape)
# change channels last to channels first format
data = moveaxis(data, 2, 0)
print(data.shape)
# change channels first to channels last format
data = moveaxis(data, 0, 2)
print(data.shape)
