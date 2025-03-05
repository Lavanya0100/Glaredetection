import cv2
import numpy as np
import glob
import os

def remove_glare_and_restore_color(image):
    """Removes glare, restores natural colors, and enhances brightness with advanced processing."""

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Multi-stage glare detection (Adaptive Threshold + Morphology)
    glare_mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY, blockSize=15, C=8)
    
    kernel = np.ones((7, 7), np.uint8)  # Larger kernel for better glare refinement
    glare_mask = cv2.morphologyEx(glare_mask, cv2.MORPH_CLOSE, kernel, iterations=5)  # Refine mask

    # Inpainting to remove glare
    glare_removed = cv2.inpaint(image, glare_mask, inpaintRadius=7, flags=cv2.INPAINT_TELEA)

    # Convert to LAB color space for color correction
    lab = cv2.cvtColor(glare_removed, cv2.COLOR_BGR2LAB)

    # Apply CLAHE to enhance brightness and contrast
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])

    # Convert back to BGR
    enhanced_image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Apply Non-Local Means Denoising for sharper, clearer images
    final_output = cv2.fastNlMeansDenoisingColored(enhanced_image, None, 10, 10, 7, 21)

    return final_output

def display_images(image_list):
    """Displays images one by one with interactive navigation."""
    if not image_list:
        print("❌ No images to display!")
        return

    for i, (original, processed, category) in enumerate(image_list):
        if original is None or processed is None:
            print(f"⚠ Error: Image {i+1} in {category} could not be loaded properly.")
            continue

        # Stack images for comparison
        stacked = np.hstack((original, processed))

        # Resize to fit screen
        scale_percent = 75  # Resize scale for visibility
        width = int(stacked.shape[1] * scale_percent / 100)
        height = int(stacked.shape[0] * scale_percent / 100)
        stacked_resized = cv2.resize(stacked, (width, height), interpolation=cv2.INTER_CUBIC)

        # Display the images
        cv2.imshow(f"{category} - {i+1}/{len(image_list)} | Press any key for next", stacked_resized)

        key = cv2.waitKey(0)  # Wait for user input
        if key == 27:  # Press ESC to exit
            break

    cv2.destroyAllWindows()  # Close all windows after displaying

# Dataset directory
dataset_dir = "C:/Users/lavgu/OneDrive/Desktop/glarefreeinput"

# Categories to process
categories = ["daytime", "nighttime", "sunrise"]
image_list = []  # Store images for display

# Load and process images
for category in categories:
    folder_path = os.path.join(dataset_dir, category)
    image_paths = glob.glob(os.path.join(folder_path, "."))  # Load all images

    if not image_paths:
        print(f"⚠ No images found in {category} folder!")
        continue

    print(f"📷 Processing {len(image_paths)} images in {category}...")

    for img_path in image_paths:
        image = cv2.imread(img_path)

        if image is None:
            print(f"❌ Error loading image: {img_path}")
            continue  # Skip invalid images

        # Process glare removal & brightness enhancement
        enhanced_image = remove_glare_and_restore_color(image)

        # Store images for display
        image_list.append((image, enhanced_image, category))

# Display images one by one
display_images(image_list)

print("✅ Process completed! All images displayed.")
