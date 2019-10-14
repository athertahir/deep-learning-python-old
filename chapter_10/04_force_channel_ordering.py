# %%
'''
## Force Channel Ordering
Finally, the channel ordering can be forced for a specific program. This can be achieved by
calling the set image dim ordering() function on the Keras backend to either ‘th’ (Theano)
for channel-first ordering, or ‘tf’ (TensorFlow) for channel-last ordering. This can be useful
if you want a program or model to operate consistently regardless of Keras default channel
ordering configuration
'''

# %%
'''
Running the example first forces channels-first ordering, then channels-last ordering, confirming each configuration by printing the channel ordering mode after the change.
'''

# %%
# force a channel ordering
from keras import backend
# force channels-first ordering
backend.set_image_dim_ordering('th')
print(backend.image_data_format())
# force channels-last ordering
backend.set_image_dim_ordering('tf')
print(backend.image_data_format())