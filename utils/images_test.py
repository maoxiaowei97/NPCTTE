from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
import os
# gen_images = gen_steps[-1]
gen_images = np.ones([5, 3, 20, 20]) * -1
gen_images[:, :, 5, 5] = 0.5
gen_images[:, :, 0, 0] = -0.5
gen_images[:, :, 18, 18] = 0.2
num_channel = gen_images.shape[1]
for i in tqdm(range(2, 3), desc='Drawing generated images'):
    plt.figure(figsize=(num_channel / 2 * 5, 5))
    for c in range(num_channel):
        plt.subplot(2, num_channel, c + 1)
        plt.title(f'Generated channel {c + 1}')
        plt.imshow(gen_images[i][c])  # [20, 20]

        plt.subplot(2, num_channel, c + 1 + num_channel)
        plt.title(f'Real channel {c + 1}')
        plt.imshow(gen_images[i][c])
    plt.savefig(os.path.join('/data/maodawei/DOT/data', 'test_images', f'test_%03d.png' % i), bbox_inches='tight')
    plt.close()