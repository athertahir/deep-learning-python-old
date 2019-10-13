# example of cropping an image
from PIL import Image
from IPython.display import display # to display images

# load image
image = Image.open('opera_house.jpg')
# create a cropped image
cropped = image.crop((100, 100, 200, 200))
# show cropped image
# cropped.show()
display(cropped)
