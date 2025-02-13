# LAB color space ensures that color remains unchanged while adjusting brightness.
# Histogram equalization redistributes pixel intensities, enhancing dark areas.
# Adding brightness shifts lighting towards a morning effect without overexposing.

import cv2
import numpy as np
import os

# Load image
image_path = "Image3.jpg"  # Replace with your image path
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert to LAB color space
lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

# Split LAB channels
l, a, b = cv2.split(lab)

# Apply Histogram Equalization on L channel
l = cv2.equalizeHist(l)

# Increase brightness by adding a constant value
l = cv2.add(l, 50)  # Adjust the value based on brightness requirement

# Merge back the LAB channels
lab_adjusted = cv2.merge((l, a, b))

# Convert back to RGB
final_image = cv2.cvtColor(lab_adjusted, cv2.COLOR_LAB2RGB)

# Generate new filename
filename, ext = os.path.splitext(image_path)
new_filename = f"{filename}_LABadjusted{ext}"

# Save the final image
cv2.imwrite(new_filename, cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR))

print(f"Saved as: {new_filename}")