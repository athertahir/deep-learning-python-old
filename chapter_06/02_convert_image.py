# %%
'''
## How to Convert an Image With Keras
Keras provides the img to array() function for converting a loaded image in PIL format into
a NumPy array for use with deep learning models. The API also provides the array to img()
function that can be used for converting a NumPy array of pixel data into a PIL image. This
can be useful if the pixel data is modified in array format because it can be saved or viewed.
The example below loads the test image, converts it to a NumPy array, and then converts it
back into a PIL image.
'''

# %%
'''
Running the example first loads the photograph in PIL format, then converts the image
to a NumPy array and reports the data type and shape. We can see that the pixel values are
converted from unsigned integers to 32-bit floating point values, and in this case, converted to
the array format [height, width, channels]. Finally, the image is converted back into PIL format.
'''

# %%
# example of converting an image with the Keras API
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img
# load the image
img = load_img('bondi_beach.jpg')
print(type(img))
# convert to numpy array
img_array = img_to_array(img)
print(img_array.dtype)
print(img_array.shape)
# convert back to image
img_pil = array_to_img(img_array)
print(type(img))