# %%
'''
## OpenCV
It is a modestly complex classifier that has also been tweaked and refined over the last
nearly 20 years. A modern implementation of the Classifier Cascade face detection algorithm is
provided in the OpenCV library. This is a C++ computer vision library that provides a Python
interface. The benefit of this implementation is that it provides pre-trained face detection
models, and provides an interface to train a model on your own dataset. OpenCV can be
installed by the package manager system on your platform, or via pip; for example:

sudo pip install opencv-python

'''

# %%
'''
Once the installation process is complete, it is important to confirm that the library was
installed correctly. This can be achieved by importing the library and checking the version
number; for example:
'''

# %%
# check opencv version
import cv2
# print version number
print(cv2.__version__)