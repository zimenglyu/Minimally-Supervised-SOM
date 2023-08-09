import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# Let's assume we have a sample data
n = 5
data = np.random.rand(2*n - 1, 2*n - 1)

# Split the data into grayscale and blue components
grayscale_data = np.copy(data)
blue_data = np.zeros_like(data)

# Fill the blue data on [2a, 2b] indices and set the same indices to 0 on grayscale data
for a in range(n):
    for b in range(n):
        blue_data[2*a, 2*b] = data[2*a, 2*b]
        grayscale_data[2*a, 2*b] = 0

# Define a colormap that maps values to grayscale and another one to blue scale
cmap_gray = mcolors.LinearSegmentedColormap.from_list('grayscale', ['lightgray', 'white'], N=256)
cmap_blue = mcolors.LinearSegmentedColormap.from_list('bluescale', ['black', 'blue'], N=256)

# Plot grayscale and blue data
plt.imshow(blue_data, cmap=cmap_blue, interpolation='nearest')
plt.imshow(grayscale_data, cmap=cmap_gray, interpolation='nearest', alpha=0.3)


# Add a colorbar for reference
plt.colorbar(label='Value')

# Display the plot
plt.show()

# import matplotlib.pyplot as plt
# import matplotlib.colors as mcolors
# import numpy as np

# # Let's assume we have a sample data
# n = 3
# data = np.random.rand(2*n - 1, 2*n - 1)

# # Define a colormap that maps values to grayscale
# cmap_gray = mcolors.LinearSegmentedColormap.from_list('grayscale', ['black', 'white'], N=256)

# # Plot data
# plt.imshow(data, cmap=cmap_gray, interpolation='nearest')

# # Draw red squares around cells with index [2a, 2b]
# for a in range(n):
#     for b in range(n):
#         plt.gca().add_patch(plt.Rectangle((2*b - 0.5, 2*a - 0.5), 1, 1, fill=False, edgecolor='blue', lw=2))

# # Add a colorbar for reference
# plt.colorbar(label='Value')

# # Display the plot
# plt.show()
