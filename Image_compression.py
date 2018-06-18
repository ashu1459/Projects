
# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import cv2
from sklearn.cluster import KMeans
import os

file_size = os.path.getsize('./im.jpg')
print('Original File Size: ', file_size, 'bytes')

file_size_dict = {}
file_size_dict[0] = file_size

im = cv2.imread('./im.jpg')
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

out_r = 100
print('Original Shape: ', im.shape)
# r=row, c=column, ch=channel
r,c,ch = im.shape 

im = cv2.resize(im, (int(out_r*float(c)/r), out_r))
r_new,c_new,ch_new = im.shape 

#print('Resized Image Shape: ', im.shape)

pixels = im.reshape((-1, 3))
#print(pixels.shape)

plt.imshow(im)
plt.show()

km = KMeans(n_clusters=8)
km.fit(pixels)
labels = km.fit_predict(pixels)

centr_colors = np.array(km.cluster_centers_, dtype='uint8')
labels = np.array(labels)
quantized = centr_colors[labels]

# print(centr_colors.dtype)
# print(centr_colors.shape)

quant_image = quantized.reshape((r_new, c_new, 3))
quant_image = cv2.cvtColor(quant_image, cv2.COLOR_RGB2BGR)

# plt.imshow(im)
# plt.show()
# plt.imshow(quant_image)
# plt.show()

fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True)
ax1.imshow(im)
ax2.imshow(quant_image)
plt.show()

# cv2.imshow("image", np.hstack([im, quant_image]))

def compress_image(K):
    km = KMeans(n_clusters=K)
    km.fit(pixels)
    labels = km.fit_predict(pixels)

    centr_colors = np.array(km.cluster_centers_, dtype='uint8')
    labels = np.array(labels)
    quantized = centr_colors[labels]

#     print(labels.shape)
#     print(quantized.shape)
    
    quant_image = quantized.reshape((r_new, c_new, 3))
    quant_image = cv2.cvtColor(quant_image, cv2.COLOR_RGB2BGR)
    
    ## Display image inline below
#     fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True)
#     fig.suptitle('K=' + str(K))
#     ax1.imshow(im)
#     ax2.imshow(quant_image)
#     plt.show()

    ## write images to disk
    img_name = './' + str(K) + '.png'
    cv2.imwrite(img_name, quant_image)
    file_size_dict[K] = os.path.getsize(img_name)

def generate_power_list(base, pwr_from, pwr_to):
    return [base**j for j in range(pwr_from, pwr_to + 1)]
    
# for values of K to be between 16 to 2048
for i in range(4, 12):
    K=2**i
    compress_image(K)
    print('Compression Ratio percent for K=', K, ' is ', (float(file_size_dict[K])/file_size_dict[0]) * 100)

freq = np.array(np.unique(km.labels_, return_counts=True)[1], dtype='float32')
print(freq)
print(freq.sum())
freq = freq/pixels.shape[0]
print(freq)
print(freq.sum())

dom = [[freq[ix], centr_colors[ix]] for ix in range(km.n_clusters)]

DOM = sorted(dom, key=lambda z:z[0], reverse=True)
#DOM = np.array(DOM)
print(DOM[0][1])
#print DOM.shape

for ix in DOM:
    print(ix)
    print("----------")

	
patch = np.zeros((50, 500, 3))
start = 0
for ix in range(km.n_clusters):
    width = int(DOM[ix][0]*patch.shape[1])
    end = start+width
    patch[:,start:end,:] = 255 - DOM[ix][1]
    start = end
    plt.axis("off")
#     plt.imshow(patch)

cv2.imwrite('color_img.jpg', patch)

# cv2.imshow('Color image', patch)
# print(patch)
# plt.show()

# calculate compression ratio percent
for i in range(4, 12):
    K=2**i
    img_name = './' + str(K) + '.png'
    new_file_size = (float(os.path.getsize(img_name))/file_size) * 100
    
    print('Compression Ratio percent for K=', K, ' is ', ("%.2f" % new_file_size), '%')
    

