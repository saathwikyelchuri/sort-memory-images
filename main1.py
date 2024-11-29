import os
from deepface import DeepFace
import shutil
from sklearn.cluster import KMeans
import numpy as np
import json
import random

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

    # Randomly select a subset of images for the memory (can adjust the number to select)
    num_images_to_select = min(5, len(memory_images))  # Choose up to 5 images (adjustable)
    selected_images = random.sample(memory_images, num_images_to_select)

    print("\nSelected Images for this Memory:")
    for i, img in enumerate(selected_images):
        print(f"{i + 1}. {os.path.basename(img)}")

    # Option to let the user manually select photos from the cluster
    selected_images_manually = []
    while True:
        select_more = input("\nDo you want to manually select any other images? (y/n): ")
        if select_more.lower() == 'y':
            # Allow user to choose photos manually by entering their indices
            print("\nEnter the numbers of the images you want to select, separated by commas:")
            print("Available images to choose from:")
            for i, img in enumerate(memory_images):
                print(f"{i + 1}. {os.path.basename(img)}")
            selected_indices = input("Enter image numbers (e.g., 1, 2, 4): ").split(",")
            for index in selected_indices:
                try:
                    selected_images_manually.append(memory_images[int(index.strip()) - 1])
                except (ValueError, IndexError):
                    print(f"Invalid index: {index.strip()}")
        elif select_more.lower() == 'n':
            break
        else:
            print("Invalid choice. Please enter 'y' or 'n'.")

    # Combine automatically selected and manually selected images
    final_selected_images = set(selected_images + selected_images_manually)

    # Create a memory folder
    memory_folder = os.path.join(output_folder, f"Memory_{cluster_idx + 1}")
    os.makedirs(memory_folder, exist_ok=True)

    # Create a JSON file for each memory with titles and descriptions
    memory_data = {
        "title": memory_title,
        "description": memory_description,
        "dominant_emotion": max(set(memory_emotions), key=memory_emotions.count),
        "images": [os.path.basename(img) for img in final_selected_images]
    }

    # Save memory data to a JSON file
    with open(os.path.join(memory_folder, "memory_data.json"), "w") as json_file:
        json.dump(memory_data, json_file, indent=4)

    # Copy selected images to memory folder
    for img_path in final_selected_images:
        shutil.copy(img_path, os.path.join(memory_folder, os.path.basename(img_path)))

print(f"Images sorted into {num_clusters} clusters, memories created, and saved in '{output_folder}'.")
