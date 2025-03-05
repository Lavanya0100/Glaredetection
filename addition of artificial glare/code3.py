import os
import cv2
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate
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

# Load images from dataset
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

    # Add streaks (light rays)
    num_streaks = random.randint(3, 6)
    for _ in range(num_streaks):
        x1, y1 = random.randint(0, w//2), random.randint(0, h//2)
        x2, y2 = random.randint(w//2, w), random.randint(h//2, h)
        cv2.line(glare, (x1, y1), (x2, y2), (1, 1, 1), thickness=random.randint(3, 8))

    # Add bright spots
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

# U-Net Model for Glare Removal
def unet_model(input_shape=(128, 128, 3)):
    inputs = Input(input_shape)

    # Encoder
    c1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    p1 = MaxPooling2D((2, 2), padding='same')(c1)

    c2 = Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
    p2 = MaxPooling2D((2, 2), padding='same')(c2)

    c3 = Conv2D(128, (3, 3), activation='relu', padding='same')(p2)
    p3 = MaxPooling2D((2, 2), padding='same')(c3)

    # Bottleneck
    b = Conv2D(256, (3, 3), activation='relu', padding='same')(p3)

    # Decoder with Skip Connections
    u1 = UpSampling2D((2, 2))(b)
    c4 = Conv2D(128, (3, 3), activation='relu', padding='same')(u1)
    merge1 = Concatenate()([c4, c3])

    u2 = UpSampling2D((2, 2))(merge1)
    c5 = Conv2D(64, (3, 3), activation='relu', padding='same')(u2)
    merge2 = Concatenate()([c5, c2])

    u3 = UpSampling2D((2, 2))(merge2)
    c6 = Conv2D(32, (3, 3), activation='relu', padding='same')(u3)
    merge3 = Concatenate()([c6, c1])

    outputs = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(merge3)

    model = Model(inputs, outputs)
    return model

# Compile the model
model = unet_model()
model.compile(optimizer='adam', loss='mse')

# Train the model (Only 4 epochs to save time)
model.fit(train_glared, train_images, epochs=4, batch_size=8, validation_data=(test_glared, test_images))

# Function to remove glare
def remove_glare(image):
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    restored = model.predict(image)[0]  # Remove glare
    return restored

# Show a few test results
num_samples = 3
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
