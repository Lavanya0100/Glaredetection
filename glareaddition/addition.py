import cv2
import numpy as np
import glob
import os

# ✅ Define Paths
sunrise_folder = "C:/Users/lavgu/OneDrive/Desktop/glarefreeinput/sunrise"
daytime_folder = "C:/Users/lavgu/OneDrive/Desktop/glarefreeinput/daytime"
output_folder = "C:/Users/lavgu/OneDrive/Desktop/glarefreeinput/output_restored"

os.makedirs(output_folder, exist_ok=True)

# ✅ Get Image Paths (Support .jpg & .jpeg)
sunrise_images = sorted(glob.glob(os.path.join(sunrise_folder, "*.jpg")) + glob.glob(os.path.join(sunrise_folder, "*.jpeg")))
daytime_images = sorted(glob.glob(os.path.join(daytime_folder, "*.jpg")) + glob.glob(os.path.join(daytime_folder, "*.jpeg")))

# ✅ Find Minimum Image Count to Prevent Errors
num_images = min(len(sunrise_images), len(daytime_images))

if num_images == 0:
    print("❌ Error: No matching images found in one or both folders.")
    exit()

if len(sunrise_images) != len(daytime_images):
    print(f"⚠️ Warning: Unequal images detected! Processing {num_images} matched pairs.")

# ✅ Function: Extract Glare from Sunrise Image
def extract_glare(sunrise_img):
    gray = cv2.cvtColor(sunrise_img, cv2.COLOR_BGR2GRAY)  # Convert to Grayscale
    _, mask = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)  # Detect bright areas
    mask = cv2.GaussianBlur(mask, (15, 15), 0)  # Smooth edges
    return mask

# ✅ Process Each Matched Pair
for i in range(num_images):
    sunrise_path = sunrise_images[i]
    daytime_path = daytime_images[i]

    sunrise_img = cv2.imread(sunrise_path)
    daytime_img = cv2.imread(daytime_path)

    # Resize to Match Dimensions
    if sunrise_img.shape[:2] != daytime_img.shape[:2]:
        daytime_img = cv2.resize(daytime_img, (sunrise_img.shape[1], sunrise_img.shape[0]))

    # Extract Glare Mask & Convert to 3 Channels
    glare_mask = extract_glare(sunrise_img)
    glare_mask_3ch = cv2.merge([glare_mask] * 3)

    # Blend Glare Onto Daytime Image
    blended_img = cv2.addWeighted(daytime_img, 1.0, glare_mask_3ch, 0.3, 0)

    # Save the Result as .jpg
    output_filename = os.path.splitext(os.path.basename(daytime_path))[0] + ".jpg"
    output_path = os.path.join(output_folder, output_filename)
    cv2.imwrite(output_path, blended_img)

print(f"✅ Glare overlay completed! {num_images} images processed. Check output in: {output_folder}")
