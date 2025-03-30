import cv2
import os
import numpy as np

def blend_images(daytime_path, sunrise_path, output_path, alpha=0.5):
    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)
    
    # Get list of images
    daytime_images = sorted(os.listdir(daytime_path))
    sunrise_images = sorted(os.listdir(sunrise_path))
    
    for day_img_name, sun_img_name in zip(daytime_images, sunrise_images):
        day_img_path = os.path.join(daytime_path, day_img_name)
        sun_img_path = os.path.join(sunrise_path, sun_img_name)
        
        # Read images
        day_img = cv2.imread(day_img_path)
        sun_img = cv2.imread(sun_img_path)
        
        # Resize to match dimensions
        sun_img = cv2.resize(sun_img, (day_img.shape[1], day_img.shape[0]))
        
        # Blend images (weighted sum)
        blended = cv2.addWeighted(day_img, 1 - alpha, sun_img, alpha, 0)
        
        # Save result
        output_file = os.path.join(output_path, day_img_name)
        cv2.imwrite(output_file, blended)
        print(f"Saved: {output_file}")

# Paths to image folders
daytime_folder = "C:/Users/lavgu/OneDrive/Desktop/glarefreeinput/daytime"
sunrise_folder = "C:/Users/lavgu/OneDrive/Desktop/glarefreeinput/sunrise"
output_folder = "C:/Users/lavgu/OneDrive/Desktop/glarefreeinput/output"

# Run blending function
blend_images(daytime_folder, sunrise_folder, output_folder, alpha=0.5)
