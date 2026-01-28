import os
import numpy as np
from PIL import Image
import glob
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def load_aloi_images(data_path="data/grey4", num_folders=10, specific_folders=None):
    # Get all available folders
    all_folders = sorted([f for f in os.listdir(data_path) 
                         if os.path.isdir(os.path.join(data_path, f))])
    
    # Select folders based on parameters
    if specific_folders is not None:
        # Validate that all specified folders exist
        missing_folders = [f for f in specific_folders if f not in all_folders]
        if missing_folders:
            raise ValueError(f"Folders not found: {missing_folders}. Available folders: {all_folders[:10]}...")
        selected_folders = specific_folders
        print(f"Loading specific folders: {selected_folders}")
    else:
        # Use first N folders
        selected_folders = all_folders[:num_folders]
        # print(f"Loading first {num_folders} folders: {selected_folders}")
    
    all_images = []
    object_ids = []
    
    for folder in selected_folders:
        folder_path = os.path.join(data_path, folder)
        image_files = sorted(glob.glob(os.path.join(folder_path, "*.png")))
        
        print(f"Loading {len(image_files)} images from folder {folder}")
        
        for img_path in image_files:
            # Load and convert to grayscale
            img = Image.open(img_path)
            if img.mode != 'L':
                img = img.convert('L')
            
            # Convert to numpy array
            img_array = np.array(img, dtype=np.float32)
            all_images.append(img_array)
            object_ids.append(folder)
    
    return np.array(all_images), object_ids


def images_to_vectors(images):
    return images.reshape(images.shape[0], -1)


def normalize_vectors(vectors, method='minmax'):
    if method == 'minmax':
        min_vals = vectors.min(axis=1, keepdims=True)
        max_vals = vectors.max(axis=1, keepdims=True)
        return (vectors - min_vals) / (max_vals - min_vals + 1e-8)
    
    elif method == 'zscore':
        mean_vals = vectors.mean(axis=1, keepdims=True)
        std_vals = vectors.std(axis=1, keepdims=True)
        return (vectors - mean_vals) / (std_vals + 1e-8)
    
    elif method == 'l2':
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / (norms + 1e-8)
    
    else:
        raise ValueError(f"Unknown method: {method}")


def plot_pca_by_object(vectors, object_ids, n_components=2, figsize=(10, 8)):
    # Perform PCA
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(vectors)
    
    # Get unique object IDs and create color map
    unique_objects = sorted(list(set(object_ids)))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_objects)))
    color_map = dict(zip(unique_objects, colors))
    
    # Create the plot
    plt.figure(figsize=figsize)
    
    for obj_id in unique_objects:
        # Get indices for this object
        mask = np.array(object_ids) == obj_id
        points = pca_result[mask]
        
        plt.scatter(points[:, 0], points[:, 1], 
                   c=[color_map[obj_id]], 
                   label=f'Object {obj_id}', 
                   alpha=0.7, s=50)
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.title('ALOI Images: PCA Plot by Object ID')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Print PCA statistics
    print(f"\nPCA Statistics:")
    print(f"Total variance explained by first {n_components} components: {pca.explained_variance_ratio_[:n_components].sum():.1%}")
    print(f"Individual component variances:")
    for i, var_ratio in enumerate(pca.explained_variance_ratio_[:n_components]):
        print(f"  PC{i+1}: {var_ratio:.1%}")


if __name__ == "__main__":
    # Example 1: Load images from first 10 folders
    print("=== Example 1: Loading first 10 folders ===")
    images, object_ids = load_aloi_images(num_folders=10)
    
    print(f"\nLoaded {images.shape[0]} images")
    print(f"Image shape: {images.shape[1:]} (height, width)")
    
    # Convert to vectors
    vectors = images_to_vectors(images)
    print(f"Vector shape: {vectors.shape}")
    print(f"Vector dimension: {vectors.shape[1]}")
    
    # Normalize vectors
    normalized_vectors = normalize_vectors(vectors, method='minmax')
    print(f"Normalized vector range: [{normalized_vectors.min():.3f}, {normalized_vectors.max():.3f}]")
    
    # Show some statistics
    print(f"\nStatistics:")
    print(f"Total images: {len(images)}")
    print(f"Images per object: 24")
    print(f"Number of objects: {len(set(object_ids))}")
    print(f"Memory usage: {vectors.nbytes / 1024 / 1024:.2f} MB")
    
    # Show first few object IDs
    unique_objects = list(set(object_ids))
    print(f"Object IDs: {unique_objects[:5]}...")
    
    # Create PCA plot colored by object_id
    print(f"\nCreating PCA visualization...")
    plot_pca_by_object(normalized_vectors, object_ids)
    
    # Example 2: Load specific folders
    print("\n\n=== Example 2: Loading specific folders ===")
    try:
        specific_images, specific_object_ids = load_aloi_images(specific_folders=['0001', '0005', '0010'])
        print(f"Loaded {specific_images.shape[0]} images from specific folders")
        print(f"Object IDs: {list(set(specific_object_ids))}")
    except ValueError as e:
        print(f"Error: {e}")
        print("Note: Make sure the ALOI dataset is available in the data/grey4 directory")
