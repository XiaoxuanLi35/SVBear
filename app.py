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

# 确保目录存在
os.makedirs('/tmp/database', exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Add parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
scikit_image_dir = os.path.join(parent_dir, 'scikit-image')
sys.path.append(scikit_image_dir)
sys.path.append(current_dir)  # 添加当前目录到Python路径

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", os.path.join('/tmp', 'uploads'))
DATABASE_PATH = os.getenv("DATABASE_PATH", os.path.join('/tmp', 'database'))
LOCAL_TEMP_DIR = os.getenv("LOCAL_TEMP_DIR", os.path.join('/tmp', 'temp_files'))
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Create directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(LOCAL_TEMP_DIR, exist_ok=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_relative_path(absolute_path):
    """Convert absolute path to relative path for URL generation"""
    try:
        rel_path = os.path.relpath(absolute_path, current_dir)
        return rel_path.replace('\\', '/')
    except ValueError:
        return absolute_path.replace('\\', '/')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)


@app.route('/image/<path:filename>')
def serve_image(filename):
    try:
        logger.debug(f"Serving image request: {filename}")

        # 获取GCS配置
        bucket_name = os.getenv('GCS_BUCKET_NAME')
        gcs_prefix = os.getenv('GCS_DATABASE_PREFIX', '')

        # 检查是否是GCS路径
        if bucket_name and (filename.startswith(gcs_prefix) or
                            not os.path.exists(os.path.join(current_dir, filename))):
            # 初始化GCS客户端
            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)

            # 准备临时文件
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as temp:
                temp_path = temp.name

            # 尝试下载文件
            try:
                blob = bucket.blob(filename)
                blob.download_to_filename(temp_path)

                # 发送文件并在请求结束后删除
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
            # 本地文件处理（保留原有逻辑）
            if os.path.isabs(filename):
                filepath = filename
            else:
                filepath = os.path.join(current_dir, filename)

            if not os.path.exists(filepath):
                logger.error(f"Image not found: {filepath}")
                return "Image not found", 404

            directory = os.path.dirname(filepath)
            filename = os.path.basename(filepath)
            return send_from_directory(directory, filename)
    except Exception as e:
        logger.error(f"Error serving image {filename}: {str(e)}\n{traceback.format_exc()}")
        return str(e), 404


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
                # 保存上传的文件
                filename = os.path.join(UPLOAD_FOLDER, file.filename)
                logger.debug(f"Saving file to: {filename}")
                file.save(filename)

                # 导入onlyedge模块
                logger.debug(f"Python path: {sys.path}")
                logger.debug(f"Current directory: {os.getcwd()}")

                try:
                    # 首先尝试原始导入
                    from Object_Recognition.onlyedge import find_top_matches
                    logger.debug("成功导入原始Object_Recognition.onlyedge模块")
                except ImportError:
                    # 如果失败，尝试从同一目录导入
                    try:
                        from onlyedge import find_top_matches
                        logger.debug("成功从本地目录导入onlyedge模块")
                    except ImportError as e:
                        logger.error(f"无法导入find_top_matches函数: {str(e)}")
                        return jsonify({'error': 'Module import error'}), 500

                # 查找匹配
                logger.debug("Finding matches...")
                matches = find_top_matches(filename, DATABASE_PATH)
                logger.debug(f"Found {len(matches)} matches")

                # 准备结果
                results = []

                # 获取GCS配置
                bucket_name = os.getenv('GCS_BUCKET_NAME')
                gcs_prefix = os.getenv('GCS_DATABASE_PREFIX', '')

                for match_name, distance in matches:
                    if bucket_name:
                        # 使用GCS路径
                        gcs_match_path = f"{gcs_prefix}{match_name}"
                        image_path = f'/image/{gcs_match_path}'
                    else:
                        # 使用本地路径
                        match_path = os.path.join(DATABASE_PATH, match_name)
                        if os.path.exists(match_path):
                            url_path = match_path.replace('\\', '/')
                            image_path = f'/image/{url_path}'
                        else:
                            # 如果文件不存在，跳过
                            continue

                    # 转换距离为相似度分数
                    similarity = 1 - distance
                    # 移除文件扩展名以用于显示
                    display_name = os.path.splitext(match_name)[0]

                    results.append({
                        'filename': display_name,
                        'similarity': float(similarity),
                        'image_path': image_path
                    })

                # 获取上传文件的URL
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
    # 验证配置
    logger.info(f"Upload folder: {UPLOAD_FOLDER}")
    logger.info(f"Database path: {DATABASE_PATH}")

    # 检查GCS配置
    bucket_name = os.getenv('GCS_BUCKET_NAME')
    if bucket_name:
        logger.info(f"Using GCS bucket: {bucket_name}")
        gcs_prefix = os.getenv('GCS_DATABASE_PREFIX', '')
        if gcs_prefix:
            logger.info(f"GCS prefix: {gcs_prefix}")
    else:
        logger.warning("GCS_BUCKET_NAME not set, using local filesystem only")

    # 运行应用，设置host='0.0.0.0'以允许外部访问
    app.run(host='0.0.0.0', port=5000, debug=False)