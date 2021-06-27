from numpy import load
import matplotlib.pyplot as plt

data = load('teddyGAN.npz')

dataA, dataB = data['arr_0'], data['arr_1']

n_samples = 4

for i in range(n_samples):
    plt.subplot(2, n_samples, 1 + i)
    plt.axis('off')
    plt.imshow(dataA[i].astype('uint8'))
for i in range(n_samples):
    plt.subplot(2, n_samples, 1 + n_samples + i)
    plt.axis('off')
    plt.imshow(dataB[i].astype('uint8'))
plt.show()