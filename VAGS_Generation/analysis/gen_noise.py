import numpy as np
import matplotlib.pyplot as plt

# Generate 3-channel RGB random noise (values between 0 and 1)
noise = np.random.rand(512, 512, 3)

# Save as PNG
plt.imsave('random_noise_rgb.png', noise)