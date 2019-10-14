# %%
'''
## Keras Channel Ordering
The Keras deep learning library is agnostic to how you wish to represent images in either channel
first or last format, but the preference must be specified and adhered to when using the library.
Keras wraps a number of mathematical libraries, and each has a preferred channel ordering.
The three main libraries that Keras may wrap and their preferred channel ordering are listed
below:
- TensorFlow: Channels last order.
- Theano: Channels first order.
- CNTK: Channels last order.

By default, Keras is configured to use TensorFlow and the channel ordering is also by default
channels last. You can use either channel ordering with any library and the Keras library. Some
libraries claim that the preferred channel ordering can result in a large difference in performance.
For example, use of the MXNet mathematical library as the backend for Keras recommends
using the channels first ordering for better performance.
'''

# %%
'''
Running the example prints your preferred channel ordering as configured in your Keras
configuration file. In this case, the channels last format is used.
'''

# %%
# show preferred channel order
from keras import backend
print(backend.image_data_format())