import cv2
import os

# Path to input and output folders
input_folder = "Objects"         # Folder containing resized object images
output_folder = "Keypoints"      # Folder to save images with keypoints drawn

# Ensure output folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Initialize SIFT
sift = cv2.SIFT_create()

# Process each image in the input folder
for image_name in os.listdir(input_folder):

    # Skip non-image files
    if not image_name.lower().endswith(".jpg"):
        print(f"Skipping non-JPG file: {image_name}")
        continue

    image_path = os.path.join(input_folder, image_name)
    
    # Load image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Could not load {image_name}. Skipping.")
        continue
    
    # Detect keypoints and descriptors
    keypoints, descriptors = sift.detectAndCompute(img, None)
    print(f"{image_name}: Detected {len(keypoints)} keypoints")
    
    # Draw keypoints on the image
    img_with_keypoints = cv2.drawKeypoints(
        img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, color=(0, 255, 0)
    )
    
    # Save the output image
    output_path = os.path.join(output_folder, f"keypoints_{image_name}")
    cv2.imwrite(output_path, img_with_keypoints)

print(f"Keypoint detection complete. Results saved in {output_folder}.")


