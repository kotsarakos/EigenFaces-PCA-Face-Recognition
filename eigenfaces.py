# M4ML Exercise: EigenFaces - Unified Python Script
# Harokopio University - Winter Semester 2025-26
#
# Implementation of EigenFaces algorithm for face recognition
# using PCA (Principal Component Analysis).
# Konstantinos Kotsaras (2022050)

import numpy as np
import cv2
import os
from glob import glob
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity


H, W = 112, 92
D = H * W      # Image Vector Dimension (10304) = 112 x 92
NumImages_PerClass = 10
TotalClasses = 40
TotalImages = NumImages_PerClass * TotalClasses
t_value_default = 0.7 # Split ratio for training
base_dir = 'pgm_faces' # Directory with PGM photos
output_dir = 'eigenfaces_output'
os.makedirs(output_dir, exist_ok=True)
print(f"Dimensions: {H}x{W}, Total Images: {TotalImages}")


def transform_images(base_dir, output_dir):
    """
    Task 1.1 
    Transforms PGM images to PNG format and saves them.
    """
    png_dir = os.path.join(output_dir, 'png_faces')
    os.makedirs(png_dir, exist_ok=True)
    
    # Iterate through all 40 class folders
    for class_id in range(1, TotalClasses + 1):
        class_folder = f's{class_id}'
        class_path = os.path.join(base_dir, class_folder)
        output_class_path = os.path.join(png_dir, class_folder)
        os.makedirs(output_class_path, exist_ok=True)
        
        # Convert each PGM file in the folder
        for pgm_file in glob(os.path.join(class_path, '*.pgm')):
            img = cv2.imread(pgm_file, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                png_file = os.path.join(output_class_path, os.path.basename(pgm_file).replace('.pgm', '.png'))
                cv2.imwrite(png_file, img)
    print("Task 1.1\nConversion from PGM to PNG completed.")
    return png_dir

def store_database(png_dir):
    """
    Task 1.2 
    Stores the image database in 4D, 3D, and 2D formats, along with Labels.
    """
    print("\n--- 1.2 Storing Database (4D, 3D, 2D) and Labels ---")
    
    # Initialization
    database_4D = np.zeros((H, W, NumImages_PerClass, TotalClasses)) # H x W x 10 x 40
    database_3D = np.zeros((D, NumImages_PerClass, TotalClasses)) # D x 10 x 40
    database_2D = np.zeros((D, TotalImages)) # D x 400 
    labels = np.zeros(TotalImages, dtype=int)
    
    image_counter = 0
    for class_id in range(1, TotalClasses + 1):
        class_folder = f's{class_id}'
        class_path = os.path.join(png_dir, class_folder)
        
        for image_index in range(1, NumImages_PerClass + 1):
            image_filename = f'{image_index}.png'
            image_path = os.path.join(class_path, image_filename)
            
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            
            if img is not None and img.shape == (H, W):
                # Store in 4D
                database_4D[:, :, image_index-1, class_id-1] = img
                
                # Flatten image to vector (10304 x 1)
                img_vector = img.flatten().reshape(D, 1)
                
                # Store in 3D
                database_3D[:, image_index-1, class_id-1] = img_vector.flatten()
                
                # Store in 2D
                database_2D[:, image_counter] = img_vector.flatten()
                
                # Store Labels (Classes 1 to 40)
                labels[image_counter] = class_id
                
                image_counter += 1
            else:
                print(f"   Warning: Image {image_path} failed to load or has incorrect dimensions.")
                
    # Save the databases and labels
    np.save(os.path.join(output_dir, 'database_4D.npy'), database_4D)
    np.save(os.path.join(output_dir, 'database_3D.npy'), database_3D)
    np.save(os.path.join(output_dir, 'database_2D.npy'), database_2D)
    np.save(os.path.join(output_dir, 'labels.npy'), labels)
    
    print("Storage for Task 1.2 completed.")
    return database_2D, labels


def partition_data(database_2D, labels, t):
    """
    Task 2.1 
    Splits the 2D database and labels into Training and Test sets using ratio t.
    """

    # Data preparation: X (features: 400x10304), y (labels: 400)
    X = database_2D.T 
    y = labels
    
    # Use StratifiedShuffleSplit to ensure the ratio t is maintained within each class
    sss = StratifiedShuffleSplit(n_splits=1, test_size=1-t, random_state=42)
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
    # Revert to D x N format (10304 x N)
    train_faces = X_train.T
    test_faces = X_test.T
    
    # Save Training and Test sets
    np.save(os.path.join(output_dir, 'train_faces.npy'), train_faces)
    np.save(os.path.join(output_dir, 'test_faces.npy'), test_faces)
    np.save(os.path.join(output_dir, 'train_labels.npy'), y_train)
    np.save(os.path.join(output_dir, 'test_labels.npy'), y_test)
    
    print(f"Task 2.1 Split completed.\nTraining: {train_faces.shape[1]} images, Test: {test_faces.shape[1]} images.")
    return train_faces, test_faces, y_train, y_test


def compute_mean_face(train_faces):
    """
    Task 3.1 
    Computes the Mean Face vector from the Training set and displays it as an image.
    """
    
    # mean_face_vector: 10304 x 1
    mean_face_vector = np.mean(train_faces, axis=1, keepdims=True)
    
    # Reshape to matrix for visualization (112 x 92)
    mean_face_image = mean_face_vector.reshape(H, W)
    
    # Display Mean Face
    plt.figure(figsize=(4, 5))
    plt.imshow(mean_face_image, cmap='gray')
    plt.title("3.1 Mean Face")
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, '3_1_Mean_Face.png'))
    plt.show()
    
    return mean_face_vector

def normalize_data(train_faces, test_faces, mean_face_vector):
    """
    Task 3.2
    Normalizes data by subtracting the Mean Face.
    """
    
    # Normalization
    centered_train_faces = train_faces - mean_face_vector
    centered_test_faces = test_faces - mean_face_vector
    
    # Store before and after normalization 
    # The 'train_faces' and 'test_faces' variables hold the 'before' data.
    np.save(os.path.join(output_dir, 'centered_train_faces.npy'), centered_train_faces)
    np.save(os.path.join(output_dir, 'centered_test_faces.npy'), centered_test_faces)
    
    print("Task 3.2 Normalization completed.")
    return centered_train_faces, centered_test_faces


def compute_eigenfaces(centered_train_faces, p_values):
    """
    Task 4.1 
    Computes EigenFaces (Eigenvectors) using SVD and visualizes the first four.
    """
    
    # The columns of U are the EigenFaces (eigenvectors of A * A^T)
    # U: D x N (10304 x 280 for t=0.7)
    U, S, V_T = np.linalg.svd(centered_train_faces, full_matrices=False)
    eigenvectors = U
    
    # Display the first 4 EigenFaces
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle("4.1 The First 4 EigenFaces")
    for i in range(4):
        eigenface = eigenvectors[:, i].reshape(H, W)
        axes[i].imshow(eigenface, cmap='gray')
        axes[i].set_title(f'EigenFace {i+1}')
        axes[i].axis('off')
    plt.savefig(os.path.join(output_dir, '4_1_First_4_EigenFaces.png'))
    plt.show()
    
    # Save all eigenvectors
    np.save(os.path.join(output_dir, 'eigenvectors.npy'), eigenvectors)
    
    print("Task 4.1 EigenFaces computation complete.")
    return eigenvectors


def compute_recognition_rate(dist_matrix, test_labels, train_labels):
    """Helper function to calculate the recognition rate from the distance matrix."""
    # Find the index of the closest training face for each test face
    closest_train_indices = np.argmin(dist_matrix, axis=1)
    
    # The predicted label is the label of the closest training face
    predicted_labels = train_labels[closest_train_indices]
    
    # Calculate accuracy
    correct_identifications = np.sum(predicted_labels == test_labels)
    total_test_faces = len(test_labels)
    recognition_rate = correct_identifications / total_test_faces
    
    return recognition_rate

def compute_and_plot_distance_matrix(centered_train_faces, centered_test_faces, eigenvectors, t, p=50):
    """
    Task 5.1 
    Computes and plots the distance matrix (Dist) using Euclidean and Cosine distances.
    """
    
    # Projection Matrix W: Eigenvectors selected (D x p)
    W = eigenvectors[:, :p]
    W_T = W.T # Projection Matrix (p x D)
    
    # Project data into the p-dimensional eigenface space
    transformed_train = W_T @ centered_train_faces
    transformed_test = W_T @ centered_test_faces
    
    # i) Euclidean distance (Dist_Euclidean: Test vs Training)
    # Input data needs to be in (N_samples x N_features) format for sklearn, so transpose is necessary
    Dist_Euclidean = euclidean_distances(transformed_test.T, transformed_train.T)
    
    # ii) Cosine distance (1 - Cosine Similarity)
    Dist_Cosine = 1 - cosine_similarity(transformed_test.T, transformed_train.T)
    
    # Plotting (Heatmaps)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    plt.sca(axes[0])
    plt.imshow(Dist_Euclidean, cmap='Blues', interpolation='nearest', aspect='auto')
    plt.colorbar(label='Euclidean Distance')
    axes[0].set_title(f'5.1 Euclidean Distance Matrix (p={p}, t={t})')
    axes[0].set_xlabel('Training Faces Index')
    axes[0].set_ylabel('Test Faces Index')
    
    plt.sca(axes[1])
    plt.imshow(Dist_Cosine, cmap='Blues', interpolation='nearest', aspect='auto')
    plt.colorbar(label='Cosine Distance (1 - Similarity)')
    axes[1].set_title(f'5.1 Cosine Distance Matrix (p={p}, t={t})')
    axes[1].set_xlabel('Training Faces Index')
    axes[1].set_ylabel('Test Faces Index')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'5_1_Distance_Matrices_p{p}.png'))
    plt.show()

    return Dist_Euclidean, Dist_Cosine

def recognition_sweep(database_2D, labels, p_values, t_values):
    """
    Task 5.2 
    Provides the Recognition Rate by varying t and p using Euclidean distance (20 results).
    """
    
    results = {}
    
    # Perform a sweep over t values
    for t in t_values:
        # Re-partition the data for the new t ratio
        train_faces, test_faces, train_labels, test_labels = partition_data(database_2D, labels, t=t)
        
        # Re-compute Mean Face and Centering
        mean_face_vector = compute_mean_face(train_faces)
        centered_train_faces, centered_test_faces = normalize_data(train_faces, test_faces, mean_face_vector)
        
        # Re-compute Eigenvectors (SVD)
        U, S, V_T = np.linalg.svd(centered_train_faces, full_matrices=False)
        eigenvectors = U
        
        results[t] = {}
        
        # Perform a sweep over p values
        for p in p_values:
            # Select p EigenFaces (W)
            W = eigenvectors[:, :p]
            W_T = W.T
            
            # Project data
            transformed_train = W_T @ centered_train_faces
            transformed_test = W_T @ centered_test_faces
            
            # Euclidean Distance
            dist_matrix = euclidean_distances(transformed_test.T, transformed_train.T)
            
            # Recognition Rate
            rate = compute_recognition_rate(dist_matrix, test_labels, train_labels)
            results[t][p] = rate
            
    # Display results
    print("\nRecognition Rate Results (Euclidean Distance):")
    p_list = p_values
    header = ["t"] + [f"p={p}" for p in p_list]
    print(f"{' | '.join(header)}")
    print("-" * (10 + len(p_list) * 7))

    for t in t_values:
        rate_list = [results[t][p] for p in p_list]
        row_output = [f"{t:.1f}"] + [f"{rate:.4f}" for rate in rate_list]
        print(f"{' | '.join(row_output)}")

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    for t, p_rates in results.items():
        p_data = list(p_rates.keys())
        rate_data = list(p_rates.values())
        ax.plot(p_data, rate_data, marker='o', label=f't={t}')
        
    ax.set_title('5.2 Recognition Rate vs p for different t (Euclidean Distance)')
    ax.set_xlabel('Number of EigenFaces (p)')
    ax.set_ylabel('Recognition Rate')
    ax.legend(title='Training Ratio (t)')
    ax.grid(True)
    plt.savefig(os.path.join(output_dir, '5_2_Recognition_Rate_Sweep.png'))
    plt.show()
    
    return results

# --- 6. FACE RECONSTRUCTION ---

def reconstruct_face(F_centered, W, F_mean):
    """Helper function to reconstruct the initial face from its centered vector and EigenFaces."""
    W_T = W.T
    
    # 1. Projection: F_transformed = W_T * F_centered
    F_transformed = W_T @ F_centered # p x 1
    
    # 2. Reconstruction: F_reconstructed = F_mean + W * F_transformed
    F_reconstructed_centered = W @ F_transformed # 10304 x 1
    F_reconstructed_init = F_mean + F_reconstructed_centered # 10304 x 1
    
    return F_reconstructed_init

def reconstruction_error(test_faces, mean_face_vector, eigenvectors, p_values):
    """
    Task 6.1 
    Provides the Total Reconstruct Error varying p.
    """

    results = {}
    
    F_init_test = test_faces # Original Test Faces (D x 120)
    F_centered_test = F_init_test - mean_face_vector # Centered Test Faces (D x 120)
    
    for p in p_values:
        if p > eigenvectors.shape[1]:
            # Skip if p is larger than the number of available eigenvectors (max 280 for t=0.7)
            continue
            
        W = eigenvectors[:, :p] # EigenFaces matrix (D x p)
        
        total_error = 0
        for j in range(F_init_test.shape[1]): # Iterate over all Test faces
            F_init_j = F_init_test[:, j:j+1]
            F_centered_j = F_centered_test[:, j:j+1]
            
            # Reconstruct the face
            F_reconstructed_init_j = reconstruct_face(F_centered_j, W, mean_face_vector)
            
            # Calculate error: ||F_init - F_hat_init||_F^2
            error_j = np.sum((F_init_j - F_reconstructed_init_j)**2) 
            total_error += error_j
            
        results[p] = total_error
        
    # Display results
    print("Total Reconstruct Error Results (t=0.7):")
    for p, error in results.items():
        print(f"   p={p}: Error = {error:.2f}")

    # Plotting
    p_list = list(results.keys())
    error_list = list(results.values())
    
    plt.figure(figsize=(10, 6))
    plt.plot(p_list, error_list, marker='o')
    plt.title('6.1 Total Reconstruct Error vs Number of EigenFaces (p)')
    plt.xlabel('Number of EigenFaces (p)')
    plt.ylabel('Total Reconstruct Error (Frobenius Norm Squared)')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, '6_1_Reconstruct_Error_Sweep.png'))
    plt.show()
    
    return results

def visualize_reconstruction(test_faces, mean_face_vector, eigenvectors, p_values, test_labels):
    """
    Task 6.2 
    Visualizes one reconstructed Test face for the first 5 persons across different p values.
    """

    # Find the index of the first test image for the first 5 classes (Persons 1 to 5)
    person_indices = []
    
    for class_id in range(1, 6): 
        # Get the index of the first occurrence of this class ID in the test labels
        first_index = np.where(test_labels == class_id)[0]
        if len(first_index) > 0:
            person_indices.append(first_index[0])
            
    if not person_indices:
        print("   Error: Could not find test samples for the first 5 persons.")
        return
        
    num_persons = len(person_indices)
    
    # Setup subplot figure
    # Rows: Original + p_values (5)
    fig, axes = plt.subplots(len(p_values) + 1, num_persons, figsize=(4 * num_persons, 4 * (len(p_values) + 1)))
    fig.suptitle('6.2 Test Face Reconstruction vs p', fontsize=16)

    F_centered_test = test_faces - mean_face_vector
    
    # Row 0: Original Test Faces
    for i, test_idx in enumerate(person_indices):
        original_face = test_faces[:, test_idx].reshape(112, 92)
        axes[0, i].imshow(original_face, cmap='gray')
        axes[0, i].set_title(f'P{i+1} (Class {test_labels[test_idx]}) - Original')
        axes[0, i].axis('off')

    # Subsequent Rows: Reconstructed Faces for each p value
    for row_idx, p in enumerate(p_values):
        if p > eigenvectors.shape[1]:
            continue
            
        W = eigenvectors[:, :p] # EigenFaces matrix (D x p)
        
        for col_idx, test_idx in enumerate(person_indices):
            F_centered_j = F_centered_test[:, test_idx:test_idx+1]
            F_reconstructed_init_j = reconstruct_face(F_centered_j, W, mean_face_vector)
            
            
            reconstructed_image = F_reconstructed_init_j.reshape(112, 92)
            
            axes[row_idx + 1, col_idx].imshow(reconstructed_image, cmap='gray')
            axes[row_idx + 1, col_idx].set_title(f'p={p}')
            axes[row_idx + 1, col_idx].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_dir, '6_2_Reconstructed_Faces.png'))
    plt.show()
    
    print("   Visualization complete.")

# ==============================================================================
# --- MAIN EXECUTION BLOCK ---
# ==============================================================================

# 1. Load and Store Data
png_dir = transform_images(base_dir, output_dir)
database_2D, labels = store_database(png_dir)

# Define p and t value sets for sweeps
p_values_recog = [2, 5, 20, 30, 50]         # For Recognition Rate (5.2)
p_values_recon = [2, 5, 20, 30, 50, 100, 300, 500, 1000] # For Reconstruction Error (6.1)
p_values_vis = [2, 20, 50, 100, 500]       # For Reconstruction Visualization (6.2)
t_values_recog = [0.2, 0.5, 0.7, 0.9]       # For Recognition Rate (5.2)

# --- Perform Steps 2, 3, 4, 5.1, 6.1, 6.2 for t=0.7 (Base Case) ---

# 2. Partition Data (t=0.7)
train_faces, test_faces, train_labels, test_labels = partition_data(database_2D, labels, t=t_value_default)

# 3. Center Data
mean_face_vector = compute_mean_face(train_faces)
centered_train_faces, centered_test_faces = normalize_data(train_faces, test_faces, mean_face_vector)

# 4. Compute EigenFaces
eigenvectors = compute_eigenfaces(centered_train_faces, p_values_recog)

# 5.1 Compute and Plot Distance Matrices (t=0.7, p=50)
Dist_Euc, Dist_Cos = compute_and_plot_distance_matrix(centered_train_faces, centered_test_faces, eigenvectors, t=t_value_default, p=50)

# 6.1 Compute Total Reconstruct Error (t=0.7)
reconstruct_error_results = reconstruction_error(test_faces, mean_face_vector, eigenvectors, p_values_recon)

# 6.2 Visualize Face Reconstruction (t=0.7)
visualize_reconstruction(test_faces, mean_face_vector, eigenvectors, p_values_vis, test_labels)

# --- Perform Step 5.2 Sweep ---

# 5.2 Recognition Rate Sweep (t=[0.2, 0.5, 0.7, 0.9], p=[2, 5, 20, 30, 50])
recognition_results = recognition_sweep(database_2D, labels, p_values_recog, t_values_recog)

print("\n--- All 10 mandatory tasks completed and results/plots saved to 'eigenfaces_output' directory. ---")