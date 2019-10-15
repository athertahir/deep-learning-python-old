# %%
'''
## Force Channel Ordering
Finally, the channel ordering can be forced for a specific program. This can be achieved by
calling the set_image_data_format function on the Keras backend to either
for channel-first ordering, or channel-last ordering. This can be useful
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
backend.set_image_data_format('channels_first')
print(backend.image_data_format())
# force channels-last ordering
backend.set_image_data_format('channels_last')
print(backend.image_data_format())