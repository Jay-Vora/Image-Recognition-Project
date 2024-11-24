import cv2
import os
import pandas as pd


def load_object_descriptors(objects_folder, sift):
    """
    Load object images, detect keypoints and descriptors, and store them in a dictionary.
    """
    object_descriptors = {}
    for obj_img_name in os.listdir(objects_folder):
        if obj_img_name.lower().endswith(".jpg"):
            obj_path = os.path.join(objects_folder, obj_img_name)
            obj_img = cv2.imread(obj_path, cv2.IMREAD_GRAYSCALE)
            keypoints, descriptors = sift.detectAndCompute(obj_img, None)
            object_descriptors[obj_img_name.split(".")[0]] = descriptors
    return object_descriptors


def match_objects_in_scene(scene_path, object_descriptors, sift, bf, threshold=10):
    """
    Match object descriptors with the scene image descriptors to detect objects.
    """
    detected_objects = []
    scene_img = cv2.imread(scene_path, cv2.IMREAD_GRAYSCALE)
    if scene_img is None:
        print(f"Failed to load scene image: {scene_path}")
        return detected_objects

    scene_keypoints, scene_descriptors = sift.detectAndCompute(scene_img, None)

    for obj_name, obj_descriptors in object_descriptors.items():
        matches = bf.knnMatch(obj_descriptors, scene_descriptors, k=2)

        # Apply Lowe's ratio test
        good_matches = [
            m for m, n in matches if m.distance < 0.75 * n.distance
        ]

        # Check match threshold
        if len(good_matches) > threshold:
            detected_objects.append(obj_name)
    return detected_objects


def calculate_metrics(true_objects, detected_objects, total_objects):
    """
    Calculate TP, FP, TN, FN, Precision, Recall, F1-Score, and Accuracy for a scene.
    """
    TP = len(set(true_objects) & set(detected_objects))
    FP = len(set(detected_objects) - set(true_objects))
    FN = len(set(true_objects) - set(detected_objects))
    TN = total_objects - (TP + FP + FN)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (TP + TN) / total_objects

    return TP, FP, TN, FN, precision, recall, f1_score, accuracy


def write_results_to_file(scene_name, detected_objects, output_file):
    """
    Write detected objects for a scene to the `testing.txt` file.
    """
    with open(output_file, "a") as f:
        f.write(f"Scene: {scene_name}\n")
        f.write(f"Detected Objects: {', '.join(detected_objects) if detected_objects else 'None'}\n\n")


def write_metrics_to_excel(results, excel_file):
    """
    Write results including TP, FP, TN, FN, and performance metrics to an Excel file.
    """
    df = pd.DataFrame(results)
    df.to_excel(excel_file, index=False)

def main():
    # Initialize paths and objects
    objects_folder = "Objects2"
    scenes_folder = "Scenes2"
    output_file = "testing.txt"
    excel_file = "Results.xlsx"

    # Ensure the output file is empty
    open(output_file, "w").close()

    # Initialize SIFT and BFMatcher
    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher()

    # Load descriptors for all objects
    object_descriptors = load_object_descriptors(objects_folder, sift)

    # Ground truth for scenes (replace with actual ground truth values)
    scene_ground_truth = {
        "S1_front.jpg": ["O1"],
        "S1_left.jpg": ["O1"],
        "S1_right.jpg": ["O1"],
        "S2_front.jpg": ["O1", "O2"],
        "S2_left.jpg": ["O1", "O2"],
        "S2_right.jpg": ["O1", "O2"],
        "S3_front.jpg": ["O1", "O2", "O3"],
        "S3_left.jpg": ["O1", "O2", "O3"],
        "S3_right.jpg": ["O1", "O2", "O3"],
        "S4_front.jpg": ["O1", "O2", "O3", "O4"],
        "S4_left.jpg": ["O1", "O2", "O3", "O4"],
        "S4_right.jpg": ["O1", "O2", "O3", "O4"],
        "S5_front.jpg": ["O1", "O2", "O3", "O4", "O5"],
        "S5_left.jpg": ["O1", "O2", "O3", "O4", "O5"],
        "S5_right.jpg": ["O1", "O2", "O3", "O4", "O5"],
        "S6_front.jpg": ["O1", "O2", "O3", "O4", "O5", "O6"],
        "S6_left.jpg": ["O1", "O2", "O3", "O4", "O5", "O6"],
        "S6_right.jpg": ["O1", "O2", "O3", "O4", "O5", "O6"],
        "S7_front.jpg": ["O1", "O2", "O3", "O4", "O5", "O6", "O7"],
        "S7_left.jpg": ["O1", "O2", "O3", "O4", "O5", "O6", "O7"],
        "S7_right.jpg": ["O1", "O2", "O3", "O4", "O5", "O6", "O7"],
        "S8_front.jpg": ["O1", "O2", "O3", "O4", "O5", "O6", "O7", "O8"],
        "S8_left.jpg": ["O1", "O2", "O3", "O4", "O5", "O6", "O7", "O8"],
        "S8_right.jpg": ["O1", "O2", "O3", "O4", "O5", "O6", "O7", "O8"],
        "S9_front.jpg": ["O1", "O2", "O3", "O4", "O5", "O6", "O7", "O8", "O9"],
        "S9_left.jpg": ["O1", "O2", "O3", "O4", "O5", "O6", "O7", "O8", "O9"],
        "S9_right.jpg": ["O1", "O2", "O3", "O4", "O5", "O6", "O7", "O8", "O9"],
        "S10_front.jpg": ["O1", "O2", "O3", "O4", "O5", "O6", "O7", "O8", "O9", "O10"],
        "S10_left.jpg": ["O1", "O2", "O3", "O4", "O5", "O6", "O7", "O8", "O9", "O10"],
        "S10_right.jpg": ["O1", "O2", "O3", "O4", "O5", "O6", "O7", "O8", "O9", "O10"]
    }

    # Initialize results container
    results = {
        "Scene": [],
        "TP": [],
        "FP": [],
        "TN": [],
        "FN": [],
        "Precision": [],
        "Recall": [],
        "F1-Score": [],
        "Accuracy": []
    }

    total_objects = len(object_descriptors)

    # Process each scene
    for scene_name in os.listdir(scenes_folder):
        if scene_name.lower().endswith(".jpg"):
            scene_path = os.path.join(scenes_folder, scene_name)
            
            # Detect objects in the scene
            detected_objects = match_objects_in_scene(scene_path, object_descriptors, sift, bf)
            
            # Write detected objects to testing.txt
            write_results_to_file(scene_name, detected_objects, output_file)
            
            # Calculate metrics
            true_objects = scene_ground_truth.get(scene_name, [])
            TP, FP, TN, FN, precision, recall, f1_score, accuracy = calculate_metrics(
                true_objects, detected_objects, total_objects
            )
            
            # Store results
            results["Scene"].append(scene_name)
            results["TP"].append(TP)
            results["FP"].append(FP)
            results["TN"].append(TN)
            results["FN"].append(FN)
            results["Precision"].append(precision)
            results["Recall"].append(recall)
            results["F1-Score"].append(f1_score)
            results["Accuracy"].append(accuracy)

    # Write metrics to Excel
    write_metrics_to_excel(results, excel_file)

    print("Processing complete. Results saved to testing.txt and Results.xlsx.")

if __name__ == "__main__":
    main()