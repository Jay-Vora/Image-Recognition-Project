import cv2
import numpy as np

# Paths to your scene image and object image
scene_path = './Scenes/S1_frontal.jpg'  # Example: 'scene.jpg'
object_path = './Objects/O1.jpg'  # Example: 'object.jpg'

# Load the images
scene_img = cv2.imread(scene_path, cv2.IMREAD_GRAYSCALE)
object_img = cv2.imread(object_path, cv2.IMREAD_GRAYSCALE)

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
scene_img_clahe = clahe.apply(scene_img)

scene_img_blurred = cv2.GaussianBlur(scene_img_clahe, (5, 5), 0)

# You can also apply CLAHE to the object image
object_img_clahe = clahe.apply(object_img)

# Detect keypoints and descriptors on the CLAHE-enhanced image
scene_kp, scene_desc = sift.detectAndCompute(scene_img_blurred, None)
object_kp, object_desc = sift.detectAndCompute(object_img_clahe, None)



# Detect keypoints and descriptors for both scene and object
# scene_kp, scene_desc = sift.detectAndCompute(scene_img, None)
# object_kp, object_desc = sift.detectAndCompute(object_img, None)

# Check if descriptors are found
if scene_desc is None or object_desc is None:
    print("Descriptors not found for either the scene or the object!")
    exit()

# Initialize BFMatcher
flann = cv2.FlannBasedMatcher(cv2.NORM_L2, crossCheck=False)

# Perform KNN match for descriptors
matches = flann.knnMatch(object_desc, scene_desc, k=2)

# Apply Lowe's ratio test to filter matches
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:  # Ratio test
        good_matches.append(m)

# If good matches are found, visualize the matches
if len(good_matches) > 0:
    # Draw matches on the image
    result_img = cv2.drawMatches(object_img, object_kp, scene_img, scene_kp, good_matches, None, flags=2)

    # Show the result
    cv2.imshow("Matches", result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(f"Found {len(good_matches)} good matches.")
else:
    print("No good matches found.")
