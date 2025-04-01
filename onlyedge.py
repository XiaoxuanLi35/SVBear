import cv2
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
import tempfile
import os
import logging
from google.cloud import storage

# 配置日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# 初始化GCS客户端
def get_storage_client():
    """获取已配置的Storage客户端"""
    # 处理Render环境中的认证
    if 'GOOGLE_CREDENTIALS_JSON' in os.environ:
        # 创建临时文件存储凭据
        fd, temp_credentials_path = tempfile.mkstemp(suffix='.json')
        with os.fdopen(fd, 'w') as f:
            f.write(os.environ['GOOGLE_CREDENTIALS_JSON'])
        # 设置环境变量
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


def extract_features(image_path):
    """
    Extract image features using only edge features
    """
    try:
        # Read image with special handling for Chinese paths
        image_data = np.fromfile(str(image_path), dtype=np.uint8)
        image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")

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
        print(f"Error processing image {image_path}: {str(e)}")
        raise


def extract_features_from_data(image_data):
    """
    Extract features from image binary data instead of file path
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


def find_top_matches(query_image_path, database_path, top_n=10):
    """
    Find matches using edge features for pixel art
    """
    # 检查是否使用GCS
    bucket_name = os.getenv('GCS_BUCKET_NAME')
    gcs_prefix = os.getenv('GCS_DATABASE_PREFIX', '')

    if bucket_name:
        logger.debug(f"使用GCS存储桶: {bucket_name}/{gcs_prefix}")
        return find_top_matches_gcs(query_image_path, bucket_name, gcs_prefix, top_n)

    # 原有的本地文件系统匹配逻辑
    # Get all images in the database
    extensions = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.gif"]
    image_files = []
    for ext in extensions:
        image_files.extend(list(Path(database_path).glob(ext)))

    if not image_files:
        raise ValueError(f"No image files found in {database_path}")

    # Extract features from all images first
    all_features = []
    valid_files = []

    try:
        query_features = extract_features(query_image_path)
        all_features.append(query_features)
        valid_files.append(query_image_path)
    except Exception as e:
        print(f"Error processing query image: {e}")
        return []

    # Process database images
    for img_path in image_files:
        try:
            features = extract_features(img_path)
            all_features.append(features)
            valid_files.append(img_path)
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")
            continue

    if len(all_features) < 2:
        raise ValueError("Not enough features extracted successfully")

    # Convert to numpy array
    all_features = np.array(all_features)

    # Apply PCA (0.90 variance ratio)
    pca = PCA()
    pca.fit(all_features)
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.argmax(cumsum >= 0.90) + 1  # Dynamic component selection

    # Transform features using selected components
    pca = PCA(n_components=n_components)
    features_pca = pca.fit_transform(all_features)

    # Print PCA information
    print("\nPCA Analysis Information:")
    print(f"Original feature dimensions: {all_features.shape[1]}")
    print(f"Reduced feature dimensions: {n_components}")
    print(f"Explained variance ratio: {sum(pca.explained_variance_ratio_):.4f}")
    print("Explained variance ratio for each principal component:")
    for i, ratio in enumerate(pca.explained_variance_ratio_):
        print(f"  PC{i + 1}: {ratio:.4f}")

    # Get query and database features
    query_features_pca = features_pca[0]
    database_features_pca = features_pca[1:]

    # Calculate distances and find top matches
    distances = [euclidean(query_features_pca, feat) for feat in database_features_pca]
    top_indices = np.argsort(distances)[:top_n]

    # Return top matches with file paths and distances
    matches = [(valid_files[i + 1].name, distances[i]) for i in top_indices]
    return matches


def find_top_matches_gcs(query_image_path, bucket_name, database_prefix, top_n=10):
    """
    Find matches using edge features for pixel art from Google Cloud Storage
    """
    logger.debug(f"在GCS中查找匹配项: {bucket_name}/{database_prefix}")

    # 初始化GCS客户端
    try:
        storage_client = get_storage_client()
        bucket = storage_client.bucket(bucket_name)

        # 列出数据库中的所有图像
        blobs = list(bucket.list_blobs(prefix=database_prefix))
        logger.debug(f"在数据库中找到 {len(blobs)} 个对象")

        # 过滤只保留图像文件
        valid_blobs = []
        for blob in blobs:
            name = blob.name.lower()
            if name.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                valid_blobs.append(blob)

        if not valid_blobs:
            raise ValueError(f"没有在 gs://{bucket_name}/{database_prefix} 中找到图像文件")

        logger.debug(f"找到 {len(valid_blobs)} 个有效图像文件")

        # Extract features from all images first
        all_features = []
        valid_files = []

        try:
            # 处理查询图像
            query_features = extract_features(query_image_path)
            all_features.append(query_features)
            valid_files.append(query_image_path)
        except Exception as e:
            logger.error(f"处理查询图像时出错: {e}")
            return []

        # 处理数据库图像
        temp_dir = tempfile.mkdtemp()
        try:
            for blob in valid_blobs:
                try:
                    # 下载图像数据
                    image_data = blob.download_as_bytes()

                    # 提取特征
                    features = extract_features_from_data(image_data)
                    all_features.append(features)
                    valid_files.append(blob.name)
                except Exception as e:
                    logger.error(f"处理 {blob.name} 时出错: {e}")
                    continue
        finally:
            # 清理临时目录
            try:
                os.rmdir(temp_dir)
            except:
                pass

        if len(all_features) < 2:
            raise ValueError("没有足够的特征被成功提取")

        # 转换为numpy数组
        all_features = np.array(all_features)

        # 应用PCA (0.90 方差比)
        pca = PCA()
        pca.fit(all_features)
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        n_components = np.argmax(cumsum >= 0.90) + 1  # 动态组件选择

        # 使用选定的组件转换特征
        pca = PCA(n_components=n_components)
        features_pca = pca.fit_transform(all_features)

        # 打印PCA信息
        logger.debug(f"原始特征维度: {all_features.shape[1]}")
        logger.debug(f"降维后特征维度: {n_components}")
        logger.debug(f"解释方差比: {sum(pca.explained_variance_ratio_):.4f}")

        # 获取查询和数据库特征
        query_features_pca = features_pca[0]
        database_features_pca = features_pca[1:]

        # 计算距离并找出最佳匹配
        distances = [euclidean(query_features_pca, feat) for feat in database_features_pca]
        top_indices = np.argsort(distances)[:top_n]

        # 返回带有文件路径和距离的最佳匹配
        matches = []
        for i in top_indices:
            # 从完整路径中提取文件名
            full_path = valid_files[i + 1]
            filename = os.path.basename(full_path)
            matches.append((filename, distances[i]))

        return matches

    except Exception as e:
        logger.error(f"查找匹配时出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return []


def display_results(query_image_path, matches, database_path):
    """
    Display query image and matching results
    """
    # Read query image
    query_img = cv2.imdecode(np.fromfile(str(query_image_path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
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

    # 检查是否使用GCS
    bucket_name = os.getenv('GCS_BUCKET_NAME')
    gcs_prefix = os.getenv('GCS_DATABASE_PREFIX', '')

    # Display matching results
    for i, (name, dist) in enumerate(matches[:10], start=1):
        if bucket_name:
            # 从GCS下载图像
            storage_client = get_storage_client()
            bucket = storage_client.bucket(bucket_name)
            blob_path = f"{gcs_prefix}{name}"
            blob = bucket.blob(blob_path)

            # 下载图像数据
            try:
                image_data = blob.download_as_bytes()
                img = cv2.imdecode(np.frombuffer(image_data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            except Exception as e:
                print(f"Error downloading {name}: {e}")
                continue
        else:
            # 本地文件系统
            img_path = Path(database_path) / name
            img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)

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
    # 默认路径
    default_database = r"C:\Users\李晓璇\Desktop\Semester 4\CS6123\project_3\spriters\Mixed"
    default_query = r"C:\Users\李晓璇\Desktop\Semester 4\CS6123\project_3\scikit-image\test_dataset\test2.jpg"

    # 获取环境变量
    bucket_name = os.getenv('GCS_BUCKET_NAME')
    gcs_prefix = os.getenv('GCS_DATABASE_PREFIX', '')

    if bucket_name:
        print(f"\n使用Google Cloud Storage: gs://{bucket_name}/{gcs_prefix}")
    else:
        print(f"\n使用本地文件系统: {default_database}")

    # 获取命令行参数或使用默认值
    database_path = Path(default_database)
    query_image = default_query

    print("\nProcessing query image:", Path(query_image).name)

    matches = find_top_matches(query_image, database_path)

    print("\nTop 10 matching images:")
    distances = [dist for _, dist in matches]  # Get all distances
    max_dist = max(distances) if distances else 1
    for name, dist in matches:
        # Convert distance to similarity score (1 - normalized_distance)
        sim = 1 - (dist / max_dist) if dist > 0 else 1
        print(f"{name}: {sim:.4f}")

    # Display matching results
    display_results(query_image, matches, database_path)