# Pythonic OO way of plotting

# Import the required packages
import matplotlib.pyplot as plt
import numpy as np

# Generate the data
x = np.arange(0, 10, 0.2)
y = np.sin(x)
z = np.cos(x)

# Generate the figure and the axes
fig, axs = plt.subplots(nrows=2, ncols=1)

# On the first axis, plot the sine and label the ordinate
axs[0].plot(x, y)
axs[0].set_ylabel('Sine')

# On the second axis, plot the cosine
axs[1].plot(x, z)
axs[1].set_ylabel('Cosine')

# Display the resulting plot
plt.show()