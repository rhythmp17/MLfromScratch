import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.utils import shuffle

img = plt.imread("test.png")

def computeMeans(data):
    means = np.mean(data,axis = 0)
    return means

def distance(x,y):
    d = y-x
    d = d.reshape(3,1)
    d_t = np.transpose(d)
    distance = np.dot(d_t,d)[0][0]
    return distance

def mykmeans(img,n_clusters,epochs):
    np.random.seed(19)
    flat_img = img.reshape(-1,3)
    m,_ = flat_img.shape
    means = np.zeros((n_clusters,3))
    for i in range(n_clusters):
        random_pixels = np.random.choice(m,size = n_clusters,replace=False)
        means[i] = computeMeans(flat_img[random_pixels])
    index = np.zeros(m)
    for _ in range(epochs):
        for pixel in range(m):
            min_distance = float('inf')
            temp = None
            for i in range(n_clusters):
                dist = distance(means[i],flat_img[pixel])
                if dist < min_distance:
                    min_distance = dist
                    temp = i

            index[pixel] = temp
        for k in range(n_clusters):
            points = flat_img[index == k]
            means[k] = computeMeans(points)
    return means,index

def custom_compressImg(img,n_clusters,epochs):
    means,index = mykmeans(img,n_clusters,epochs)
    centroid = np.array(means)
    recovered = centroid[index.astype(int), :]

    # getting back the 3d matrix (row, col, rgb(3))
    recovered = recovered.reshape(img.shape)
    return recovered
    

def compare_with_sklearn(image, n_colors):
    # Convert image to 2D array (pixels by channels)
    w, h, d = original_shape = tuple(image.shape)
    image_array = np.reshape(image, (w * h, d))

    # Fit KMeans model to image data
    image_array_sample = shuffle(image_array, random_state=0)[:1000]
    kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)

    # Predict cluster assignments for all pixels in the image
    labels = kmeans.predict(image_array)

    # Replace each pixel with its cluster center
    compressed_image = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            compressed_image[i][j] = kmeans.cluster_centers_[labels[label_idx]]
            label_idx += 1

    custom_compressed = custom_compressImg(img,n_colors,5)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('Custom Compressed Image')
    plt.imshow(custom_compressed)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Compressed Image (SK-Learn) ({} colors)'.format(n_colors))
    plt.imshow(compressed_image)
    plt.axis('off')
    plt.imshow(compressed_image)
    plt.axis("off")
    plt.show()


compare_with_sklearn(img,7)

def computeMeans(data):
    means = np.mean(data,axis = 0)
    return means

def euclidean_distance(matrix_size, cell1, cell2):
    # Get the row and column numbers for each cell
    row1, col1 = (cell1 - 1) // matrix_size, (cell1 - 1) % matrix_size
    row2, col2 = (cell2 - 1) // matrix_size, (cell2 - 1) % matrix_size
    
    # Calculate the Euclidean distance
    distance = np.sqrt((row2 - row1)**2 + (col2 - col1)**2)
    
    return distance

def customDistance(x,y,i,j,spatial_weight):
    color_distance = np.sqrt(np.sum((x - y)**2))
    spatial_distance = euclidean_distance(512,i,j)
    return (color_distance) + (spatial_weight*spatial_distance)

def mykmeans_spatial(img,n_clusters,epochs):
    np.random.seed(19)
    flat_img = img.reshape(-1,3)
    m,_ = flat_img.shape
    means = np.zeros((n_clusters,3))
    pixels = np.zeros((n_clusters))
    for i in range(n_clusters):
        random_pixels = np.random.choice(m,size = n_clusters,replace=False)
        pixels[i] = np.mean(random_pixels)
        means[i] = computeMeans(flat_img[random_pixels])
    index = np.zeros(m)
    for _ in range(epochs):
        for pixel in range(m):
            min_distance = float('inf')
            temp = None
            for i in range(n_clusters):
                dist = customDistance(means[i],flat_img[pixel],pixels[i],pixel,1)
                if dist < min_distance:
                    min_distance = dist
                    temp = i

            index[pixel] = temp
        for k in range(n_clusters):
            points = flat_img[index == k]
            new_mean = np.mean(points)
            means[k] = computeMeans(points)
            pixels[k] = new_mean
    return means,index

def custom_compressImg(img,n_clusters,epochs):
    means,index = mykmeans_spatial(img,n_clusters,epochs)
    centroid = np.array(means)
    recovered = centroid[index.astype(int), :]

    # getting back the 3d matrix (row, col, rgb(3))
    recovered = recovered.reshape(img.shape)
    return recovered
    

lst = [4]
for i in lst:
    custom_compressed = custom_compressImg(img,i,5)
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(img)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('Compressed Image with Spatial Coherence ({} colors, spatial weight={})'.format(i, 1))
    plt.imshow(custom_compressed)
    plt.axis('off')
    plt.show()

