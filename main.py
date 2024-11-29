# import os
# from deepface import DeepFace
# import shutil
# from sklearn.cluster import KMeans
# import numpy as np

# # Paths
# source_folder = "D:\\hp\\photos_folder"  # Folder containing images
# output_folder = "D:\\hp\\sorted_photos"  # Folder to save sorted images

# # Check if source folder exists
# if not os.path.exists(source_folder):
#     raise ValueError(f"The folder '{source_folder}' does not exist.")

# # Get all image paths
# image_files = [os.path.join(source_folder, f) for f in os.listdir(source_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
# if not image_files:
#     raise ValueError(f"No valid image files found in '{source_folder}'.")

# # Create output folder if it doesn't exist
# os.makedirs(output_folder, exist_ok=True)

# # Analyze images and generate embeddings
# embeddings = []
# valid_image_paths = []

# print("Analyzing images...")
# for image_file in image_files:
#     try:
#         result = DeepFace.represent(img_path=image_file, enforce_detection=False, model_name="Facenet")
#         embeddings.append(result[0]["embedding"])  # Extract embedding
#         valid_image_paths.append(image_file)  # Save corresponding image path
#         print(f"Processed: {image_file}")
#     except Exception as e:
#         print(f"Error processing {image_file}: {e}")

# # Check if embeddings were generated
# if not embeddings:
#     raise ValueError("No valid embeddings were generated. Please check the images.")

# # Convert embeddings to NumPy array
# embeddings_array = np.array(embeddings)

# # Cluster embeddings using K-Means
# print("Clustering faces...")
# num_clusters = min(len(valid_image_paths), 5)  # Set number of clusters (adjustable)
# kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
# kmeans.fit(embeddings_array)

# # Create folders for each cluster and copy images
# print("Sorting images into clusters...")
# for cluster_idx in range(num_clusters):
#     cluster_folder = os.path.join(output_folder, f"Person_{cluster_idx + 1}")
#     os.makedirs(cluster_folder, exist_ok=True)

#     for i, label in enumerate(kmeans.labels_):
#         if label == cluster_idx:
#             shutil.copy(valid_image_paths[i], os.path.join(cluster_folder, os.path.basename(valid_image_paths[i])))

# print(f"Images sorted into {num_clusters} clusters and saved in '{output_folder}'.")



import os
from deepface import DeepFace
import shutil
from sklearn.cluster import KMeans
import numpy as np
import json

# Paths
source_folder = "D:\\hp\\photos_folder"  # Folder containing images
output_folder = "D:\\hp\\sorted_photos"  # Folder to save sorted images

# Check if source folder exists
if not os.path.exists(source_folder):
    raise ValueError(f"The folder '{source_folder}' does not exist.")

# Get all image paths
image_files = [os.path.join(source_folder, f) for f in os.listdir(source_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
if not image_files:
    raise ValueError(f"No valid image files found in '{source_folder}'.")

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Analyze images and generate embeddings
embeddings = []
valid_image_paths = []
emotion_labels = []

print("Analyzing images...")
for image_file in image_files:
    try:
        # Generate face embeddings using DeepFace
        result = DeepFace.represent(img_path=image_file, enforce_detection=False, model_name="Facenet")
        embeddings.append(result[0]["embedding"])  # Extract embedding
        valid_image_paths.append(image_file)  # Save corresponding image path

        # Detect emotion for each image (using DeepFace)
        emotion_result = DeepFace.analyze(img_path=image_file, actions=['emotion'], enforce_detection=False)
        emotions = emotion_result[0]["dominant_emotion"]
        emotion_labels.append(emotions)  # Save emotion labels

        print(f"Processed: {image_file}, Emotion: {emotions}")
    except Exception as e:
        print(f"Error processing {image_file}: {e}")

# Check if embeddings were generated
if not embeddings:
    raise ValueError("No valid embeddings were generated. Please check the images.")

# Convert embeddings to NumPy array
embeddings_array = np.array(embeddings)

# Cluster embeddings using K-Means
print("Clustering faces...")
num_clusters = min(len(valid_image_paths), 5)  # Set number of clusters (adjustable)
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
kmeans.fit(embeddings_array)

# Create folders for each cluster and copy images
print("Sorting images into clusters...")
for cluster_idx in range(num_clusters):
    cluster_folder = os.path.join(output_folder, f"Person_{cluster_idx + 1}")
    os.makedirs(cluster_folder, exist_ok=True)

    memory_images = []  # List to store images for this memory
    memory_emotions = []  # List to store emotions for this memory
    for i, label in enumerate(kmeans.labels_):
        if label == cluster_idx:
            img_path = valid_image_paths[i]
            shutil.copy(img_path, os.path.join(cluster_folder, os.path.basename(img_path)))
            memory_images.append(img_path)
            memory_emotions.append(emotion_labels[i])

    # Manually input title and description for each memory
    print(f"\nMemory {cluster_idx + 1} - Please enter a title and description for this memory:")
    memory_title = input(f"Enter the title for Memory {cluster_idx + 1}: ")
    memory_description = input(f"Enter the description for Memory {cluster_idx + 1}: ")

    # Create a memory folder
    memory_folder = os.path.join(output_folder, f"Memory_{cluster_idx + 1}")
    os.makedirs(memory_folder, exist_ok=True)

    # Create a JSON file for each memory with titles and descriptions
    memory_data = {
        "title": memory_title,
        "description": memory_description,
        "dominant_emotion": max(set(memory_emotions), key=memory_emotions.count),
        "images": [os.path.basename(img) for img in memory_images]
    }

    # Save memory data to a JSON file
    with open(os.path.join(memory_folder, "memory_data.json"), "w") as json_file:
        json.dump(memory_data, json_file, indent=4)

    # Copy images to memory folder
    for img_path in memory_images:
        shutil.copy(img_path, os.path.join(memory_folder, os.path.basename(img_path)))

print(f"Images sorted into {num_clusters} clusters, memories created, and saved in '{output_folder}'.")
