import cv2
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from scipy.spatial.distance import euclidean
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tempfile
import os
import logging
from google.cloud import storage
import pickle
import time

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Paths for cached features
FEATURES_CACHE_FILE = "gcs_features_cache.npz"
FILENAMES_CACHE_FILE = "gcs_filenames_cache.pkl"
PCA_MODEL_CACHE_FILE = "pca_model_cache.pkl"


# Initialize GCS client
def get_storage_client():
    """Get the configured Storage client"""
    # Handle credentials in Render environment
    if 'GOOGLE_CREDENTIALS_JSON' in os.environ:
        # Create a temporary file for credentials
        fd, temp_credentials_path = tempfile.mkstemp(suffix='.json')
        with os.fdopen(fd, 'w') as f:
            f.write(os.environ['GOOGLE_CREDENTIALS_JSON'])
        # Set environment variable
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = temp_credentials_path

    return storage.Client()


def extract_edge_features(image):
    """
    Extract edge features using rotation-sensitive gradient features.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Parameters for edge detection
    ksize = 3
    threshold1 = 100
    threshold2 = 200
    direction_bins = 18

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (ksize, ksize), 0)

    # Canny edge detection
    edges = cv2.Canny(blurred, threshold1, threshold2)

    # Calculate Sobel gradients
    sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=ksize)
    sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=ksize)

    # Calculate gradient direction and magnitude
    gradient_direction = np.arctan2(sobely, sobelx) * 180 / np.pi  # Range: [-180, 180]
    gradient_magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)

    # Create non-rotation-invariant gradient histogram
    # Use full 360-degree range instead of folding into 180 degrees
    direction_range = [-180, 180]
    direction_step = 360 / direction_bins
    direction_hist = np.zeros(direction_bins)

    for i in range(direction_bins):
        lower = direction_range[0] + i * direction_step
        upper = lower + direction_step
        mask = (gradient_direction >= lower) & (gradient_direction < upper)
        direction_hist[i] = np.sum(gradient_magnitude[mask])

    # Normalize direction histogram
    direction_hist = direction_hist / (np.sum(direction_hist) + 1e-6)

    # Calculate separate X and Y gradient features
    x_gradient_hist = np.histogram(sobelx.flatten(), bins=direction_bins // 2, range=[-1, 1])[0]
    y_gradient_hist = np.histogram(sobely.flatten(), bins=direction_bins // 2, range=[-1, 1])[0]

    # Normalize X and Y gradient histograms
    x_gradient_hist = x_gradient_hist / (np.sum(x_gradient_hist) + 1e-6)
    y_gradient_hist = y_gradient_hist / (np.sum(y_gradient_hist) + 1e-6)

    # Calculate edge density
    block_size = 8
    h, w = edges.shape
    blocks_h = h // block_size
    blocks_w = w // block_size
    density_map = np.zeros((blocks_h, blocks_w))

    for i in range(blocks_h):
        for j in range(blocks_w):
            block = edges[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size]
            density_map[i, j] = np.sum(block > 0) / (block_size * block_size)

    edge_density = density_map.flatten()

    # Combine all edge features
    return np.concatenate([
        direction_hist,  # Non-rotation-invariant gradient directions
        x_gradient_hist,  # X gradient distribution
        y_gradient_hist,  # Y gradient distribution
        edge_density  # Edge density features
    ])


def extract_features_from_data(image_data):
    """
    Extract features from image binary data
    """
    try:
        # Decode image from binary data
        image = cv2.imdecode(np.frombuffer(image_data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode image data")

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize image to 48x48
        image = cv2.resize(image, (48, 48), interpolation=cv2.INTER_AREA)

        # Extract edge features
        edge_features = extract_edge_features(image)

        # Normalize feature vector
        edge_features = edge_features / np.linalg.norm(edge_features)

        return edge_features

    except Exception as e:
        print(f"Error processing image data: {str(e)}")
        raise


def extract_all_gcs_features(bucket_name, database_prefix, force_refresh=False):
    """
    Extract features from all images in the GCS bucket and save them to disk.
    If force_refresh is False and cached features exist, load from cache instead.
    """
    # Check if cached features exist and we don't need to force refresh
    if not force_refresh and os.path.exists(FEATURES_CACHE_FILE) and os.path.exists(FILENAMES_CACHE_FILE):
        logger.info(f"Loading features from cache files")
        try:
            features = np.load(FEATURES_CACHE_FILE)['features']
            with open(FILENAMES_CACHE_FILE, 'rb') as f:
                filenames = pickle.load(f)

            # Load PCA model if available
            if os.path.exists(PCA_MODEL_CACHE_FILE):
                with open(PCA_MODEL_CACHE_FILE, 'rb') as f:
                    pca_model = pickle.load(f)
            else:
                pca_model = None

            logger.info(f"Loaded {len(features)} features and {len(filenames)} filenames from cache")
            return features, filenames, pca_model
        except Exception as e:
            logger.error(f"Error loading from cache: {e}, will extract features fresh")

    logger.info(f"Extracting features from GCS bucket: {bucket_name}/{database_prefix}")
    start_time = time.time()

    database_prefix = database_prefix.replace("'", "").replace('"', "")

    try:
        storage_client = get_storage_client()
        bucket = storage_client.bucket(bucket_name)

        blobs = list(bucket.list_blobs(prefix=database_prefix))
        logger.info(f"Found {len(blobs)} objects in database")

        valid_blobs = []
        for blob in blobs:
            name = blob.name.lower()
            if name.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                valid_blobs.append(blob)

        logger.info(f"Found {len(valid_blobs)} valid images in database")

        all_features = []
        all_filenames = []
        batch_size = 10

        for i in range(0, len(valid_blobs), batch_size):
            batch = valid_blobs[i:i + batch_size]
            logger.info(f"Processing batch {i // batch_size + 1}/{(len(valid_blobs) + batch_size - 1) // batch_size}")

            for blob in batch:
                try:
                    image_data = blob.download_as_bytes()
                    features = extract_features_from_data(image_data)
                    all_features.append(features)
                    all_filenames.append(blob.name)
                    logger.debug(f"Processed {blob.name}")
                except Exception as e:
                    logger.error(f"Error processing {blob.name}: {e}")
                    continue

            # Clean up memory periodically
            if i % (batch_size * 10) == 0 and i > 0:
                import gc
                gc.collect()

        if len(all_features) == 0:
            raise ValueError("No features extracted from any images")

        # Convert to numpy array
        all_features = np.array(all_features)

        # Create and fit PCA model
        logger.info("Fitting PCA model to features")
        pca = PCA()
        pca.fit(all_features)
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        n_components = np.argmax(cumsum >= 0.90) + 1

        pca_model = PCA(n_components=n_components)
        pca_model.fit(all_features)

        logger.info(f"Original feature dimensions: {all_features.shape[1]}")
        logger.info(f"Reduced feature dimensions: {n_components}")
        logger.info(f"Explained variance ratio: {sum(pca_model.explained_variance_ratio_):.4f}")

        # Save features and filenames to disk
        logger.info("Saving features and filenames to cache files")
        np.savez(FEATURES_CACHE_FILE, features=all_features)
        with open(FILENAMES_CACHE_FILE, 'wb') as f:
            pickle.dump(all_filenames, f)
        with open(PCA_MODEL_CACHE_FILE, 'wb') as f:
            pickle.dump(pca_model, f)

        elapsed_time = time.time() - start_time
        logger.info(f"Feature extraction completed in {elapsed_time:.2f} seconds")

        return all_features, all_filenames, pca_model

    except Exception as e:
        logger.error(f"Error extracting features: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None, None


def find_top_matches(query_image_data, bucket_name, database_prefix, top_n=10, force_refresh=False):
    """
    Find matches using edge features for pixel art from Google Cloud Storage
    Using cached features if available
    """
    try:
        # Get features for all images in database
        database_features, database_filenames, pca_model = extract_all_gcs_features(
            bucket_name, database_prefix, force_refresh)

        if database_features is None or len(database_features) == 0:
            logger.error("No database features available")
            return []

        # Extract features from query image
        try:
            query_features = extract_features_from_data(query_image_data)
        except Exception as e:
            logger.error(f"Error extracting query image features: {e}")
            return []

        # Apply PCA transformation
        if pca_model is not None:
            logger.debug("Applying PCA transformation")
            query_features_pca = pca_model.transform([query_features])[0]
            database_features_pca = pca_model.transform(database_features)
        else:
            # If no PCA model (shouldn't happen but just in case)
            logger.warning("No PCA model available, using raw features")
            query_features_pca = query_features
            database_features_pca = database_features

        # Calculate distances
        distances = [euclidean(query_features_pca, feat) for feat in database_features_pca]

        # Find top matches
        top_indices = np.argsort(distances)[:top_n]

        # Return top matches with file paths and distances
        matches = []
        for i in top_indices:
            full_path = database_filenames[i]
            filename = os.path.basename(full_path).replace("'", "").replace('"', "")
            matches.append((filename, distances[i]))

        return matches

    except Exception as e:
        logger.error(f"Error finding matches: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return []


def display_results(query_image_data, matches, bucket_name, gcs_prefix):
    """
    Display query image and matching results
    """
    # Read query image
    query_img = cv2.imdecode(np.frombuffer(query_image_data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if query_img is None:
        print("Failed to read query image")
        return

    # Convert BGR to RGB
    if len(query_img.shape) == 2:
        query_img = cv2.cvtColor(query_img, cv2.COLOR_GRAY2RGB)
    elif query_img.shape[-1] == 4:
        # Handle alpha channel
        white_bg = np.ones_like(query_img[:, :, 3], dtype=np.uint8) * 255
        alpha = query_img[:, :, 3] / 255.0
        query_img = cv2.cvtColor(query_img[:, :, :3], cv2.COLOR_BGR2RGB)
        query_img = (query_img * alpha[:, :, np.newaxis] + white_bg[:, :, np.newaxis] * (
                1 - alpha[:, :, np.newaxis])).astype(np.uint8)
    elif query_img.shape[-1] == 3:
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)

    # Resize query image to 48x48 for display
    query_img = cv2.resize(query_img, (48, 48), interpolation=cv2.INTER_AREA)

    # Calculate the number of rows and columns needed
    n_matches = min(len(matches), 10)  # Display max 10 matches
    total_images = n_matches + 1  # Including the query image
    cols = 5
    rows = (total_images + cols - 1) // cols

    # Create matplotlib figure
    plt.figure(figsize=(16, 4 * rows))
    plt.suptitle('Pixel Art Matching Results', fontsize=16, y=0.95)

    # Display query image
    plt.subplot(rows, cols, 1)
    plt.imshow(query_img, interpolation='nearest')
    plt.title('Query Image', pad=10, fontsize=12)
    plt.axis('off')

    # Get all distances for normalization
    distances = [dist for _, dist in matches]
    max_dist = max(distances) if distances else 1

    # Display matching results
    for i, (name, dist) in enumerate(matches[:10], start=1):
        clean_name = name.replace("'", "").replace('"', "")
        # download from GCS
        storage_client = get_storage_client()
        bucket = storage_client.bucket(bucket_name)
        blob_path = f"{gcs_prefix}{clean_name}"
        blob = bucket.blob(blob_path)

        # download information of the image
        try:
            image_data = blob.download_as_bytes()
            img = cv2.imdecode(np.frombuffer(image_data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        except Exception as e:
            print(f"Error downloading {name}: {e}")
            continue

        if img is None:
            continue

        # Convert BGR to RGB and resize
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[-1] == 4:
            # Handle alpha channel
            white_bg = np.ones_like(img[:, :, 3], dtype=np.uint8) * 255
            alpha = img[:, :, 3] / 255.0
            img = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB)
            img = (img * alpha[:, :, np.newaxis] + white_bg[:, :, np.newaxis] * (1 - alpha[:, :, np.newaxis])).astype(
                np.uint8)
        elif img.shape[-1] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize to 48x48
        img = cv2.resize(img, (48, 48), interpolation=cv2.INTER_AREA)

        # Display matching image
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img, interpolation='nearest')
        # Extract file name without extension
        name_without_ext = Path(name).stem
        # Convert distance to similarity score (1 - normalized_distance)
        sim = 1 - (dist / max_dist) if dist > 0 else 1
        plt.title(f'{name_without_ext}\nsim: {sim:.3f}', pad=10, fontsize=12)
        plt.axis('off')

    plt.tight_layout()
    # Adjust the spacing between subplots
    plt.subplots_adjust(wspace=0.3, hspace=0.4)
    plt.show()


if __name__ == "__main__":
    # Get environment variables
    bucket_name = os.getenv('GCS_BUCKET_NAME')
    gcs_prefix = os.getenv('GCS_DATABASE_PREFIX', '')
    force_refresh = os.getenv('FORCE_REFRESH_FEATURES', '').lower() in ('true', 'yes', '1')

    if not bucket_name:
        print("Error: GCS_BUCKET_NAME environment variable is required")
        exit(1)

    print(f"\nUsing Google Cloud Storage: gs://{bucket_name}/{gcs_prefix}")

    # Preload all features
    print("\nPreloading features from GCS bucket...")
    features, filenames, _ = extract_all_gcs_features(bucket_name, gcs_prefix, force_refresh)
    print(f"Loaded {len(features)} features from {len(filenames)} images")

    # Load query image
    query_image_path = os.getenv('QUERY_IMAGE_PATH')
    if not query_image_path:
        print("Error: QUERY_IMAGE_PATH environment variable is required")
        exit(1)

    try:
        # Read query image data
        with open(query_image_path, 'rb') as f:
            query_image_data = f.read()

        print(f"\nProcessing query image: {os.path.basename(query_image_path)}")

        # Find matches
        matches = find_top_matches(query_image_data, bucket_name, gcs_prefix)

        print("\nTop 10 matching images:")
        distances = [dist for _, dist in matches]  # Get all distances
        max_dist = max(distances) if distances else 1
        for name, dist in matches:
            # Convert distance to similarity score (1 - normalized_distance)
            sim = 1 - (dist / max_dist) if dist > 0 else 1
            print(f"{name}: {sim:.4f}")

        # Display matching results
        display_results(query_image_data, matches, bucket_name, gcs_prefix)

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback

        print(traceback.format_exc())