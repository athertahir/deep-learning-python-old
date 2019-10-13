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