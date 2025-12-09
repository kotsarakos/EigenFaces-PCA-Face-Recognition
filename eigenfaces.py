import os
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt

# Task 1.1 (Transform and Save images)

input_folder = "pgm_faces"
output_folder = "png_faces"

os.makedirs(output_folder, exist_ok=True)

person_folders = sorted(glob.glob(os.path.join(input_folder, "s*")))

for person_path in person_folders:
    person_name = os.path.basename(person_path)
    person_output = os.path.join(output_folder, person_name)
    os.makedirs(person_output, exist_ok=True)

    pgm_files = sorted(glob.glob(os.path.join(person_path, "*.pgm")))

    for pgm_path in pgm_files:
        
        img = cv2.imread(pgm_path, cv2.IMREAD_GRAYSCALE)

        
        base = os.path.basename(pgm_path).replace(".pgm", ".png")
        output_path = os.path.join(person_output, base)

        cv2.imwrite(output_path, img)

        print(f"Saved: {output_path}")

print("Task 1 completed successfully.")

# Task 1.2
# Store images and labels:
# i) 4D volume: [num_images, height, width, channels]
# ii) 3D volume: [num_images, height, width]  (for grayscale)
# iii) 2D matrix: [num_images, num_pixels]   (flattened images)
# Labels: [num_images]  (class/category for each image)


png_folder = "png_faces"
output_folder = "dataset_npy"


os.makedirs(output_folder, exist_ok=True)


person_folders = sorted(glob.glob(os.path.join(png_folder, "s*")))

# ((H)Height x (W)Width)
H, W = 112, 92
D = H * W
N_classes = 40
N_images = 10
Total_images = N_classes * N_images


# 4D VOLUME
# shape: (112, 92, 10, 40)
data_4d = np.zeros((H, W, N_images, N_classes), dtype=np.uint8)

for class_idx, person_path in enumerate(person_folders):
    png_files = sorted(glob.glob(os.path.join(person_path, "*.png")))
    
    for img_idx, img_path in enumerate(png_files):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        data_4d[:, :, img_idx, class_idx] = img

np.save(os.path.join(output_folder, "data_4d.npy"), data_4d)

# Labels for 4D
labels_4d = np.arange(1, N_classes + 1)
np.save(os.path.join(output_folder, "labels_4d.npy"), labels_4d)

print("Saved 4D dataset:", data_4d.shape)
print("Saved 4D labels:", labels_4d.shape)

# 3D VOLUME
# shape: (10304, 10, 40)
data_3d = np.zeros((D, N_images, N_classes), dtype=np.float32)

for class_idx in range(N_classes):
    for img_idx in range(N_images):
        img = data_4d[:, :, img_idx, class_idx]
        data_3d[:, img_idx, class_idx] = img.reshape(D)

np.save(os.path.join(output_folder, "data_3d.npy"), data_3d)

# Labels for 3D
labels_3d = labels_4d.copy()
np.save(os.path.join(output_folder, "labels_3d.npy"), labels_3d)

print("Saved 3D dataset:", data_3d.shape)
print("Saved 3D labels:", labels_3d.shape)


# 2D MATRIX
# shape: (10304, 400)
data_2d = np.zeros((D, Total_images), dtype=np.float32)
labels_2d = np.zeros(Total_images, dtype=np.int32)

col = 0
for class_idx in range(N_classes):
    for img_idx in range(N_images):
        vec = data_3d[:, img_idx, class_idx]
        data_2d[:, col] = vec
        labels_2d[col] = class_idx + 1
        col += 1

np.save(os.path.join(output_folder, "data_2d.npy"), data_2d)
np.save(os.path.join(output_folder, "labels_2d.npy"), labels_2d)

print("Saved 2D dataset:", data_2d.shape)
print("Saved 2D labels:", labels_2d.shape)

print("\nTask 1.2 completed successfully.")


# Task 2.1
# Store image-vectors and labels
# Split into Train (70%) and Test (30%)
# Train: 3D array [num_train, height, width]
# Test: 3D array [num_test, height, width]
# Labels: [num_images] for classes/categories

output_folder = "dataset_npy"
data_3d = np.load(os.path.join(output_folder, "data_3d.npy"))   
labels_3d = np.load(os.path.join(output_folder, "labels_3d.npy"))

t = 0.7
N_images = 10
train_count = int(t * N_images)
test_count = N_images - train_count

# 10304, 10, 40
D, _, N_classes = data_3d.shape      

# Create Train/Test
database_Train = np.zeros((D, train_count, N_classes), dtype=np.float32)
database_Test  = np.zeros((D, test_count,  N_classes), dtype=np.float32)

labels_train = []
labels_test = []


# Split per class (person)
for cls in range(N_classes):
    
    
    idx = np.random.permutation(N_images)
    
    train_idx = idx[:train_count]
    test_idx  = idx[train_count:]
    
    
    database_Train[:, :, cls] = data_3d[:, train_idx, cls]
    database_Test[:, :, cls]  = data_3d[:, test_idx,  cls]
    
    labels_train.extend([cls + 1] * train_count)
    labels_test.extend([cls + 1] * test_count)

labels_train = np.array(labels_train)
labels_test = np.array(labels_test)

# Save results
np.save(os.path.join(output_folder, "database_Train.npy"), database_Train)
np.save(os.path.join(output_folder, "database_Test.npy"),  database_Test)
np.save(os.path.join(output_folder, "labels_train.npy"), labels_train)
np.save(os.path.join(output_folder, "labels_test.npy"),  labels_test)

print("Train:", database_Train.shape, " Test:", database_Test.shape)
print("Task 2.1 completed successfully.")


# Task 3.1
# Compute the Mean Face (average of all training images)
# Show the Mean Face as an image

# Load Train database from Task 2.1
output_folder = "dataset_npy"
database_Train = np.load(os.path.join(output_folder, "database_Train.npy"))


# (10304, 7, 40)
D, train_count, N_classes = database_Train.shape
H, W = 112, 92   # original image dimensions

# Convert 3D train set to 2D (shape: 10304 x 280)
train_images_2d = database_Train.reshape(D, train_count * N_classes)

# Calculate Mean Face 
mean_face_vector = np.mean(train_images_2d, axis=1)  # shape (10304,)

# Turn into 2D image 112x92
mean_face_image = mean_face_vector.reshape(H, W)

# Show Mean Face
plt.figure(figsize=(4,4))
plt.imshow(mean_face_image, cmap='gray')
plt.title("Mean Face")
plt.axis('off')
plt.show()

# Save Mean Face Photo
np.save(os.path.join(output_folder, "mean_face.npy"), mean_face_vector)

print("Task 3.1 completed successfully.")

# Task 3.2
# Normalize data (subtract Mean Face)
# Save Train/Test sets before & after normalization

output_folder = "dataset_npy"

database_Train = np.load(os.path.join(output_folder, "database_Train.npy"))
database_Test  = np.load(os.path.join(output_folder, "database_Test.npy")) 
mean_face_vector = np.load(os.path.join(output_folder, "mean_face.npy"))      

D, train_count, N_classes = database_Train.shape
_, test_count, _ = database_Test.shape

# Prep for normalized datasets
database_Train_normalized = np.zeros_like(database_Train, dtype=np.float32)
database_Test_normalized  = np.zeros_like(database_Test, dtype=np.float32)

# Normalize Train
for cls in range(N_classes):
    for img in range(train_count):
        original_img = database_Train[:, img, cls]
        database_Train_normalized[:, img, cls] = original_img - mean_face_vector

# Normalize Test
for cls in range(N_classes):
    for img in range(test_count):
        original_img = database_Test[:, img, cls]
        database_Test_normalized[:, img, cls] = original_img - mean_face_vector


# Save normalized versions
np.save(os.path.join(output_folder, "database_Train_normalized.npy"), database_Train_normalized)
np.save(os.path.join(output_folder, "database_Test_normalized.npy"),  database_Test_normalized)

print("Train normalized:", database_Train_normalized.shape)
print("Test normalized:", database_Test_normalized.shape)
print("Task 3.2 completed successfully.")

# Task 4.1
# Compute first p eigenvectors (p=2,5,20,30,50)
# Show first 4 EigenFaces as images

output_folder = "dataset_npy"
train_norm = np.load(os.path.join(output_folder, "database_Train_normalized.npy"))

# shape = (10304, 7, 40)
D, train_count, N_classes = train_norm.shape
H, W = 112, 92


# Step 1: Reshape to 2D matrix A 
A = train_norm.reshape(D, train_count * N_classes)

print("A shape:", A.shape)


# SVD decomposition
U, S, Vt = np.linalg.svd(A, full_matrices=False)

# U shape = (10304, 280)
# S shape = (280,)
# Vt shape = (280, 280)

print("SVD done.")
print("U shape:", U.shape)


# Extract eigenfaces
# These are the columns of U
p_values = [2, 5, 20, 30, 50]

eigenfaces = {p: U[:, :p] for p in p_values}

# Save eigenfaces
np.save(os.path.join(output_folder, "eigenfaces_U.npy"), U)



# Show FIRST 4 eigenfaces as images

plt.figure(figsize=(8, 8))
for i in range(4):
    eigenface_vector = U[:, i]
    eigenface_img = eigenface_vector.reshape(H, W)  

    plt.subplot(2, 2, i+1)
    plt.imshow(eigenface_img, cmap='gray')
    plt.title(f"EigenFace {i+1}")
    plt.axis('off')

plt.tight_layout()
plt.show()

print("Task 4.1 completed successfully.")
