import numpy as np
import tifffile as tiff
import os
from glob import glob
import matplotlib.pyplot as plt
from stardist.models import StarDist2D
from csbdeep.utils import normalize
from skimage.measure import regionprops, label
from skimage.segmentation import clear_border
from skimage.transform import resize
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

def extract_cells(image, labels):
    cell_images = []
    labeled_cells = label(labels)
    labeled_cells = clear_border(labeled_cells)
    props = regionprops(labeled_cells, intensity_image=image)
    
    for prop in props:
        minr, minc, maxr, maxc = prop.bbox
        cell_image = image[minr:maxr, minc:maxc]
        cell_images.append(cell_image)
    
    return cell_images

# フォルダ内のすべての.tifファイルを取得
folder_path = "/Users/matsuokoujirou/Documents/Data/imaging_data/Luca"
file_paths = glob(os.path.join(folder_path, '*.tif'))

# Stardistモデルのロード
model = StarDist2D.from_pretrained('2D_versatile_fluo')

# すべての.tifファイルから細胞画像を抽出
all_cell_images_green = []
for file_path in file_paths:
    image = tiff.imread(file_path)
    magenta_channel = image[..., 2]
    normalized_image = normalize(magenta_channel)
    
    labels, details = model.predict_instances(normalized_image)
    
    height, width = labels.shape
    filtered_labels = np.copy(labels)
    props = regionprops(labels)
    for prop in props:
        minr, minc, maxr, maxc = prop.bbox
        if minr < 10 or minc < 10 or maxr > (height - 10) or maxc > (width - 10):
            filtered_labels[labels == prop.label] = 0
    
    green_channel = image[..., 1]
    
    cell_images_green = extract_cells(green_channel, filtered_labels)
    all_cell_images_green.extend(cell_images_green)

cell_size = (64, 64)
resized_cell_images_green = [resize(cell, cell_size, anti_aliasing=True) for cell in all_cell_images_green]

cell_images_array_green = np.array(resized_cell_images_green)

num_cells_green = cell_images_array_green.shape[0]
print(f"抽出されたgreen ch細胞画像の数: {num_cells_green}")

if num_cells_green == 0:
    raise ValueError("抽出された細胞画像がありません。データセットを確認してください。")

cell_images_array = np.expand_dims(cell_images_array_green, axis=-1)

num_images_to_display = min(10, len(resized_cell_images_green))
plt.figure(figsize=(12, 12))
for i in range(num_images_to_display):
    plt.subplot(1, num_images_to_display, i+1)
    plt.imshow(resized_cell_images_green[i], cmap='gray')
    plt.title("Green Channel")
    plt.axis('off')
plt.show()

datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

input_img = Input(shape=(64, 64, 1))

x = Conv2D(128, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder.summary()

cell_images_array = cell_images_array.astype('float32') / 255.

autoencoder.fit(datagen.flow(cell_images_array, cell_images_array, batch_size=32),
                steps_per_epoch=int(len(cell_images_array) / 32), 
                epochs=50)

encoder = Model(input_img, encoded)

encoded_images = encoder.predict(cell_images_array)

encoded_images_flat = encoded_images.reshape((encoded_images.shape[0], -1))

scaler = StandardScaler()
encoded_images_flat_scaled = scaler.fit_transform(encoded_images_flat)

oc_svm = OneClassSVM(kernel='rbf', gamma='auto').fit(encoded_images_flat_scaled)

reconstructed_images = autoencoder.predict(cell_images_array)
reconstruction_error = np.mean(np.power(cell_images_array - reconstructed_images, 2), axis=(1, 2, 3))

plt.hist(reconstruction_error, bins=50)
plt.xlabel('Reconstruction Error')
plt.ylabel('Frequency')
plt.title('Distribution of Reconstruction Error for Normal Cells')
plt.show()

threshold = np.percentile(reconstruction_error, 95)

anomalies = reconstruction_error > threshold

anomaly_rate = np.mean(anomalies)

autoencoder.save(folder_path + '/RG2_CAE_green_v6_50.h5')

print(f"異常細胞の割合: {anomaly_rate * 100:.2f}%")

# ランダムに5つの再構成画像と入力画像を表示
num_images_to_display = 5
random_indices = np.random.choice(len(cell_images_array), num_images_to_display, replace=False)
plt.figure(figsize=(12, 6))
for i, idx in enumerate(random_indices):
    plt.subplot(2, num_images_to_display, i+1)
    plt.imshow(cell_images_array[idx].reshape(64, 64), cmap='gray')
    plt.title('Original')
    plt.axis('off')
    
    plt.subplot(2, num_images_to_display, i+1+num_images_to_display)
    plt.imshow(reconstructed_images[idx].reshape(64, 64), cmap='gray')
    plt.title('Reconstructed')
    plt.axis('off')
plt.show()
