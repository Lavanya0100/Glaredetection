import os
import cv2
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from sklearn.model_selection import train_test_split

# Load dataset
def load_images_from_folder(folder, img_size=(128, 128)):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, img_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
            images.append(img)
    return np.array(images) / 255.0  # Normalize

# Load images from each category
data_dir = "C:/Users/lavgu/OneDrive/Desktop/glarefreeinput"
sunrise_images = load_images_from_folder(os.path.join(data_dir, "sunrise"))
daytime_images = load_images_from_folder(os.path.join(data_dir, "daytime"))
nighttime_images = load_images_from_folder(os.path.join(data_dir, "nighttime"))

# Combine all images
all_images = np.concatenate((sunrise_images, daytime_images, nighttime_images), axis=0)

# Split into training and testing sets
train_images, test_images = train_test_split(all_images, test_size=0.2, random_state=42)

# Function to create realistic sun glare effect
def add_glare(image):
    h, w, _ = image.shape
    glare = np.zeros_like(image, dtype=np.float32)

    # Create multiple streaks (light rays)
    num_streaks = random.randint(3, 6)
    for _ in range(num_streaks):
        x1, y1 = random.randint(0, w//2), random.randint(0, h//2)
        x2, y2 = random.randint(w//2, w), random.randint(h//2, h)
        cv2.line(glare, (x1, y1), (x2, y2), (1, 1, 1), thickness=random.randint(3, 8))

    # Create light reflections (flare spots)
    num_spots = random.randint(2, 5)
    for _ in range(num_spots):
        center_x, center_y = random.randint(0, w), random.randint(0, h)
        radius = random.randint(10, 50)
        intensity = random.uniform(0.2, 0.6)
        cv2.circle(glare, (center_x, center_y), radius, (intensity, intensity, intensity), -1)

    return np.clip(image + glare, 0, 1)  # Add glare without exceeding brightness limits

# Generate training data (glared images as input, original as output)
train_glared = np.array([add_glare(img) for img in train_images])
test_glared = np.array([add_glare(img) for img in test_images])

# Autoencoder Model
input_img = Input(shape=(128, 128, 3))
x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
output_img = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, output_img)
autoencoder.compile(optimizer='adam', loss='mse')

# Train the model with only 4 epochs to save time
autoencoder.fit(train_glared, train_images, epochs=4, batch_size=8, validation_data=(test_glared, test_images))

# Function to remove glare
def remove_glare(image):
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    restored = autoencoder.predict(image)[0]  # Remove glare
    return restored

# Show multiple test images (Original → Glared → Restored)
num_samples = 5
sample_indices = random.sample(range(len(test_images)), num_samples)

plt.figure(figsize=(12, 4 * num_samples))

for i, idx in enumerate(sample_indices):
    original = test_images[idx]
    glared = test_glared[idx]
    restored = remove_glare(glared)

    plt.subplot(num_samples, 3, i * 3 + 1)
    plt.imshow(original)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(num_samples, 3, i * 3 + 2)
    plt.imshow(glared)
    plt.title("Glared Image")
    plt.axis("off")

    plt.subplot(num_samples, 3, i * 3 + 3)
    plt.imshow(restored)
    plt.title("Restored Image")
    plt.axis("off")

plt.tight_layout()
plt.show()
