import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.utils import save_image

# ✅ Load the original image (Glare-Free)
image_path = "C:/Users/lavgu/Downloads/pexels-optical-chemist-340351297-30861272.jpg"  # Provide the correct image path
original_img = cv2.imread(image_path)
original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
original_img = cv2.resize(original_img, (256, 256))  # Resize for uniformity

# 🔹 Function to Add Artificial Sun Glare
def add_sun_glare(img):
    glare = np.zeros_like(img, dtype=np.uint8)
    center = (150, 100)  # Position of glare
    radius = 80  # Glare size
    color = (255, 255, 200)  # Yellowish-white glare
    cv2.circle(glare, center, radius, color, -1, cv2.LINE_AA)
    glare = cv2.GaussianBlur(glare, (55, 55), 50)  # Soft blur effect
    glared_img = cv2.addWeighted(img, 0.8, glare, 0.4, 0)  # Merge glare
    return glared_img

# ✅ Generate a Glared Image
glared_img = add_sun_glare(original_img)

# 🔹 Convert to Tensor
transform = transforms.ToTensor()
original_tensor = transform(original_img).unsqueeze(0)  # Shape: [1, 3, 256, 256]
glared_tensor = transform(glared_img).unsqueeze(0)

# 🔹 Display Original & Glared Images
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(original_img)
ax[0].set_title("Original Image")
ax[0].axis("off")

ax[1].imshow(glared_img)
ax[1].set_title("Glared Image")
ax[1].axis("off")
plt.show()

# 🔹 Define a Simple Autoencoder Model for Glare Removal
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=2, stride=2),
            nn.Sigmoid(),  # Normalize output to [0,1]
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# ✅ Initialize Model, Loss & Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Autoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ✅ Training the Autoencoder
epochs = 2500
glared_tensor, original_tensor = glared_tensor.to(device), original_tensor.to(device)

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(glared_tensor)
    loss = criterion(output, original_tensor)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")

# ✅ Save the Trained Model
torch.save(model.state_dict(), "glare_removal_model.pth")

# 🔹 Remove Glare from the Glared Image
model.eval()
with torch.no_grad():
    restored_img_tensor = model(glared_tensor)

# ✅ Convert Tensor to Image
restored_img = restored_img_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
restored_img = (restored_img * 255).astype(np.uint8)

# ✅ Display Results
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(original_img)
ax[0].set_title("Original Image")
ax[0].axis("off")

ax[1].imshow(glared_img)
ax[1].set_title("Glared Image")
ax[1].axis("off")

ax[2].imshow(restored_img)
ax[2].set_title("Restored Image")
ax[2].axis("off")

plt.show()

# ✅ Save Restored Image
cv2.imwrite("restored_image.jpg", cv2.cvtColor(restored_img, cv2.COLOR_RGB2BGR))

print("✅ Process Completed: Restored Image Saved as 'restored_image.jpg'")
