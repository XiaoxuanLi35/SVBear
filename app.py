from flask import Flask, render_template, request, jsonify, send_file, send_from_directory, after_this_request
import cv2
import numpy as np
from pathlib import Path
import base64
import os
import logging
import sys
import traceback
import tempfile
from google.cloud import storage
import threading
import pickle

# 确保目录存在
os.makedirs('/tmp/database', exist_ok=True)

# 配置日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# 添加父目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(current_dir)  # 添加当前目录到Python路径

app = Flask(__name__)

# 配置
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", os.path.join('/tmp', 'uploads'))
DATABASE_PATH = os.getenv("DATABASE_PATH", os.path.join('/tmp', 'database'))
LOCAL_TEMP_DIR = os.getenv("LOCAL_TEMP_DIR", os.path.join('/tmp', 'temp_files'))
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
FEATURES_CACHE_FILE = os.path.join('/tmp', "gcs_features_cache.npz")
FILENAMES_CACHE_FILE = os.path.join('/tmp', "gcs_filenames_cache.pkl")
PCA_MODEL_CACHE_FILE = os.path.join('/tmp', "pca_model_cache.pkl")
COMBINED_FEATURES_GCS_PATH = "combined_features.npy"

# 创建目录
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(LOCAL_TEMP_DIR, exist_ok=True)

# 全局变量
preloading_started = False
preloading_complete = False
cached_features = None
cached_filenames = None


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_relative_path(absolute_path):
    """转换绝对路径为相对路径用于URL生成"""
    try:
        rel_path = os.path.relpath(absolute_path, current_dir)
        return rel_path.replace('\\', '/')
    except ValueError:
        return absolute_path.replace('\\', '/')


def extract_features_from_data(image_data):
    """
    与onlyedge.py完全一致的特征提取流程
    """
    try:
        # 1. 解码图像数据
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            logger.error("Failed to decode image")
            return None

        # 2. 与第一个脚本完全一致的预处理
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR转RGB
        img = cv2.resize(img, (48, 48), interpolation=cv2.INTER_AREA)  # 固定48x48

        # 3. 特征提取
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        # Canny边缘检测
        edges = cv2.Canny(blurred, 100, 200)

        # Sobel梯度计算
        sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)

        # 方向直方图(18维)
        grad_dir = np.arctan2(sobely, sobelx) * 180 / np.pi
        grad_mag = np.sqrt(sobelx ** 2 + sobely ** 2)
        dir_hist = np.zeros(18)
        for i in range(18):
            lower = -180 + i * 20
            upper = lower + 20
            mask = (grad_dir >= lower) & (grad_dir < upper)
            dir_hist[i] = np.sum(grad_mag[mask])
        dir_hist /= (np.sum(dir_hist) + 1e-6)

        # X/Y梯度直方图(各9维)
        x_hist = np.histogram(sobelx.flatten(), bins=9, range=[-1, 1])[0]
        y_hist = np.histogram(sobely.flatten(), bins=9, range=[-1, 1])[0]
        x_hist /= (np.sum(x_hist) + 1e-6)
        y_hist /= (np.sum(y_hist) + 1e-6)

        # 边缘密度(36维)
        block_size = 8
        h, w = edges.shape
        density_map = np.zeros((h // block_size, w // block_size))
        for i in range(h // block_size):
            for j in range(w // block_size):
                block = edges[i * block_size:(i + 1) * block_size,
                        j * block_size:(j + 1) * block_size]
                density_map[i, j] = np.sum(block > 0) / (block_size ** 2)
        edge_density = density_map.flatten()

        # 合并特征(72维)
        features = np.concatenate([dir_hist, x_hist, y_hist, edge_density])

        # L2归一化
        features /= np.linalg.norm(features)

        return features

    except Exception as e:
        logger.error(f"特征提取错误: {str(e)}")
        return None


def preload_features():
    global preloading_complete, cached_features, cached_filenames

    try:
        bucket_name = os.getenv('GCS_BUCKET_NAME', 'svbearitems')
        gcs_features_path = os.getenv('GCS_FEATURES_PATH', COMBINED_FEATURES_GCS_PATH)

        if bucket_name:
            logger.info(f"Starting background preloading of features from gs://{bucket_name}/{gcs_features_path}...")

            # Initialize GCS client
            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(gcs_features_path)

            # Download the combined features file
            with tempfile.NamedTemporaryFile(delete=False) as temp:
                temp_path = temp.name
                blob.download_to_filename(temp_path)
                logger.info(f"Downloaded combined features to {temp_path}")

                # Load the combined features
                combined_data = np.load(temp_path, allow_pickle=True).item()
                cached_features = combined_data['features']
                cached_filenames = combined_data['filenames']

                logger.info(f"Loaded {len(cached_features)} feature vectors and {len(cached_filenames)} filenames")

                # Clean up temp file
                os.remove(temp_path)

            # Save to local cache for future use
            np.savez(FEATURES_CACHE_FILE, features=cached_features)
            with open(FILENAMES_CACHE_FILE, 'wb') as f:
                pickle.dump(cached_filenames, f)

            logger.info("Background preloading of features complete.")
            preloading_complete = True
        else:
            logger.warning("No GCS bucket configured, skipping preloading.")
            preloading_complete = True
    except Exception as e:
        logger.error(f"Error in preload_features: {str(e)}\n{traceback.format_exc()}")
        preloading_complete = True


@app.route('/')
def index():
    global preloading_started

    # Start preloading features in background if not already started
    if not preloading_started:
        preloading_thread = threading.Thread(target=preload_features)
        preloading_thread.daemon = True
        preloading_thread.start()
        preloading_started = True

    return render_template('index.html')


@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)


@app.route('/image/<path:filename>')
def serve_image(filename):
    try:
        logger.debug(f"Serving image request: {filename}")
        clean_filename = filename.replace("'", "").replace('"', "")

        # Get GCS configuration
        bucket_name = os.getenv('GCS_BUCKET_NAME', 'svbearitems')
        gcs_prefix = os.getenv('GCS_DATABASE_PREFIX', '')

        # Check if it is a GCS path
        if bucket_name and (filename.startswith(gcs_prefix) or
                            not os.path.exists(os.path.join(current_dir, filename))):
            # Initialize GCS client
            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)

            # Prepare temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(clean_filename)[1]) as temp:
                temp_path = temp.name

            # Try downloading file
            try:
                blob = bucket.blob(clean_filename)
                blob.download_to_filename(temp_path)

                @after_this_request
                def remove_file(response):
                    try:
                        os.remove(temp_path)
                    except Exception as e:
                        logger.error(f"Error removing temp file: {e}")
                    return response

                return send_file(temp_path, as_attachment=False)
            except Exception as e:
                logger.error(f"Error downloading from GCS: {e}")
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                raise
        else:
            # Handle local file
            if os.path.isabs(clean_filename):
                filepath = filename
            else:
                filepath = os.path.join(current_dir, clean_filename)

            if not os.path.exists(filepath):
                logger.error(f"Image not found: {filepath}")
                return "Image not found", 404

            directory = os.path.dirname(filepath)
            filename = os.path.basename(filepath)
            return send_from_directory(directory, filename)
    except Exception as e:
        logger.error(f"Error serving image {filename}: {str(e)}\n{traceback.format_exc()}")
        return str(e), 404


def find_matches_with_cached_features(query_features, top_n=10):
    """
    Find matches using pre-extracted features from cache
    """
    global cached_features, cached_filenames

    try:
        # Use global cached features if available
        if cached_features is not None and cached_filenames is not None:
            features = cached_features
            filenames = cached_filenames
        else:
            # Try to load from local cache files
            logger.debug("Loading features from local cache")
            if os.path.exists(FEATURES_CACHE_FILE) and os.path.exists(FILENAMES_CACHE_FILE):
                features = np.load(FEATURES_CACHE_FILE)['features']
                with open(FILENAMES_CACHE_FILE, 'rb') as f:
                    filenames = pickle.load(f)
            else:
                logger.error("Features cache not available")
                return []

        # Check if PCA model exists
        if os.path.exists(PCA_MODEL_CACHE_FILE):
            with open(PCA_MODEL_CACHE_FILE, 'rb') as f:
                pca_model = pickle.load(f)

                # Apply PCA transformation
                query_features_pca = pca_model.transform([query_features])[0]
                features_pca = pca_model.transform(features)
        else:
            # No PCA model, use raw features
            query_features_pca = query_features
            features_pca = features

        # Calculate distances quickly using numpy operations
        from scipy.spatial.distance import cdist
        distances = cdist([query_features_pca], features_pca, 'euclidean')[0]

        # Find top matches
        top_indices = np.argsort(distances)[:top_n]

        # Return top matches
        matches = []
        for i in top_indices:
            full_path = filenames[i]
            filename = os.path.basename(full_path).replace("'", "").replace('"', "")
            matches.append((filename, distances[i]))

        return matches

    except Exception as e:
        logger.error(f"Error finding matches with cached features: {str(e)}")
        return []


@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            logger.error("No file part in request")
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            logger.error("No selected file")
            return jsonify({'error': 'No file selected'}), 400

        if file and allowed_file(file.filename):
            try:
                # Save uploaded file
                filename = os.path.join(UPLOAD_FOLDER, file.filename)
                logger.debug(f"Saving file to: {filename}")
                file.save(filename)

                # Get GCS configuration
                bucket_name = os.getenv('GCS_BUCKET_NAME', 'svbearitems')
                gcs_prefix = os.getenv('GCS_DATABASE_PREFIX', '')

                # Read the file
                with open(filename, 'rb') as f:
                    image_data = f.read()

                # Extract features from query image
                query_features = extract_features_from_data(image_data)

                if query_features is None:
                    return jsonify({'error': 'Failed to extract features from image'}), 500

                # Find matches using cached features
                matches = find_matches_with_cached_features(query_features)
                logger.debug(f"Found {len(matches)} matches using cached features")

                results = []

                for match_name, distance in matches:
                    if bucket_name:
                        gcs_match_path = f"{gcs_prefix}{match_name}"
                        image_path = f'/image/{gcs_match_path}'
                    else:
                        match_path = os.path.join(DATABASE_PATH, match_name)
                        if os.path.exists(match_path):
                            url_path = match_path.replace('\\', '/')
                            image_path = f'/image/{url_path}'
                        else:
                            continue

                    similarity = max(0, 1 - (distance / 10))  # Normalize similarity score
                    display_name = os.path.splitext(match_name)[0]

                    results.append({
                        'filename': display_name,
                        'similarity': float(similarity),
                        'image_path': image_path
                    })

                uploaded_url = filename.replace('\\', '/')

                return jsonify({
                    'success': True,
                    'uploaded_file': f'/image/{uploaded_url}',
                    'matches': results
                })

            except Exception as e:
                logger.error(f"Error processing upload: {str(e)}\n{traceback.format_exc()}")
                return jsonify({'error': f'Error processing image: {str(e)}'}), 500

        return jsonify({'error': 'Unsupported file type'}), 400

    except Exception as e:
        logger.error(f"Upload error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    logger.info(f"Upload folder: {UPLOAD_FOLDER}")
    logger.info(f"Database path: {DATABASE_PATH}")

    bucket_name = os.getenv('GCS_BUCKET_NAME', 'svbearitems')
    if bucket_name:
        logger.info(f"Using GCS bucket: {bucket_name}")
        gcs_prefix = os.getenv('GCS_DATABASE_PREFIX', '')
        if gcs_prefix:
            logger.info(f"GCS prefix: {gcs_prefix}")
    else:
        logger.warning("GCS_BUCKET_NAME not set, using local filesystem only")

    app.run(host='0.0.0.0', port=5000, debug=False)