import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

img_path = 'data/udacity_sdc.png'

# load color image
bgr_image = cv2.imread(img_path)
# convert to grayscale
gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

# normalize, rescale the entries to lie in [0, 1]
gray_image = gray_image.astype("float32")/255

# plot image
plt.imshow(gray_image, cmap='gray')
plt.show()

# Define and visualize the filters
filter_vals = np.array([[-1, -1, 1, 1],
                        [-1, -1, 1, 1],
                        [-1, -1, 1, 1],
                        [-1, -1, 1, 1]])
print('Filter shape: ', filter_vals.shape)

# Defining four different filters, all of which are linear combinations of the 'filter_vals' above
filter_1 = filter_vals
filter_2 = -filter_1
filter_3 = filter_1.T
filter_4 = -filter_3
filters = np.array([filter_1, filter_2, filter_3, filter_4])

for i in range(4):
    print('Filter {}: \n'.format(str(i+1)), filters[i])

# Next, we initialize a convolutional layer so that it contains all the created filters. Then add a maxpooling layer,
# with a kernel size of (2x2) so you can see that the image resolution has been reduced after this step!
# A maxpooling layer reduces the x-y size of an input and only keeps the most active pixel values. Below is an example
# of a 2x2 pooling kernel, with a stride of 2, applied to a small patch of grayscale pixel values; reducing the size of
# the patch by a factor of 4. Only the maximum pixel values in 2x2 remain in the new, pooled output.
class Net(nn.Module):
    def __init__(self, weight):
        super(Net, self).__init__()

        # initialize the weights of the convolutional layer to the be weights of the 4 filters we have defined
        k_height, k_width = weight.shape[2:]

        # define the convolution layer, assume there are 4 grayscale filters
        self.conv = nn.Conv2d(in_channels=1, out_channels=4,
                              kernel_size=(k_height, k_width),
                              stride=1, padding=0, bias=False)
        self.conv.weight = nn.Parameter(weight)

        # defining the pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # calculate the output of the convolutional layer pre and post activation
        conv_x = self.conv(x)
        activated_x = F.relu(conv_x)

        # apply the pooling layer
        pooled_x = self.pool(activated_x)

        # return all 3 layers
        return conv_x, activated_x, pooled_x


# instantiate the model and set the weights
weight = torch.from_numpy(filters).unsqueeze(1).type(torch.FloatTensor)
model = Net(weight)
print(model)

# Visualize the output for each layer
def viz_layer(layer, n_filters=4):
    fig = plt.figure(figsize=(20, 20))

    for i in range(n_filters):
        ax = fig.add_subplot(1, n_filters, i+1)
        # grab layer outputs
        ax.imshow(np.squeeze(layer[0,i].data.numpy()), cmap='gray')
        ax.set_title('Output {}'.format(str(i+1)))
    plt.show()

# plot the original image
plt.imshow(gray_image, cmap='gray')

# visualize all filters
fig = plt.figure(figsize=(12, 6))
# fig.subplots_adjust(left=0, right=1.5, bottom=0.8, top=1, hspace=0.05, wspace=0.05)
for i in range(4):
    ax = fig.add_subplot(1, 4, i+1, xticks=[], yticks=[])
    ax.imshow(filters[i], cmap='gray')
    ax.set_title('Filter {}'.format(str(i+1)))
plt.show()

# convert the image into an input tensor
gray_image_tensor = torch.from_numpy(gray_image).unsqueeze(0).unsqueeze(1)

# get all the layers
conv_layer, activated_layer, pooled_layer = model(gray_image_tensor)

# visualise the output of the activated layers
viz_layer(activated_layer)

# visualise the output of the pooled layers
viz_layer(pooled_layer)

