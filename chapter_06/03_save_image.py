# %%
'''
## How to Save an Image With Keras
The Keras API also provides the save img() function to save an image to file. The function
takes the path to save the image, and the image data in NumPy array format. The file format
is inferred from the filename, but can also be specified via the file format argument. This can
be useful if you have manipulated image pixel data, such as scaling, and wish to save the image
for later use. The example below loads the photograph image in grayscale format, converts it to
a NumPy array, and saves it to a new file name.
'''

# %%
'''
Running the example first loads the image and forces the color channels to be grayscale. The
image is then converted to a NumPy array and saved to the new filename bondi beach grayscale.jpg
in the current working directory. To confirm that the file was saved correctly, it is loaded again
as a PIL image and details of the image are reported.
'''

# %%
# example of saving an image with the Keras API
from keras.preprocessing.image import load_img
from keras.preprocessing.image import save_img
from keras.preprocessing.image import img_to_array
# load image as as grayscale
img = load_img('bondi_beach.jpg', color_mode='grayscale')
# convert image to a numpy array
img_array = img_to_array(img)
# save the image with a new filename
save_img('bondi_beach_grayscale.jpg', img_array)
# load the image to confirm it was saved correctly
img = load_img('bondi_beach_grayscale.jpg')
print(type(img))
print(img.format)
print(img.mode)
print(img.size)
#img.show()

from IPython.display import display # to display images
display(img)
