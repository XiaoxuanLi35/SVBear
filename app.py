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

# Ensure directory exists
os.makedirs('/tmp/database', exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Add parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
scikit_image_dir = os.path.join(parent_dir, 'scikit-image')
sys.path.append(scikit_image_dir)
sys.path.append(current_dir)  # Add current directory to Python path

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
        clean_filename = filename.replace('"', "")
        if clean_filename.startwith("''"):
            clean_filename = clean_filename[2:]

        # Get GCS configuration
        bucket_name = os.getenv('GCS_BUCKET_NAME')
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

                # Import onlyedge module
                logger.debug(f"Python path: {sys.path}")
                logger.debug(f"Current directory: {os.getcwd()}")

                try:
                    # Attempt original import
                    from onlyedge import find_top_matches
                    logger.debug("Successfully imported Object_Recognition.onlyedge module")
                except ImportError:
                    # Attempt local import
                    try:
                        from onlyedge import find_top_matches
                        logger.debug("Successfully imported onlyedge from local directory")
                    except ImportError as e:
                        logger.error(f"Failed to import find_top_matches: {str(e)}")
                        return jsonify({'error': 'Module import error'}), 500

                # Find matches
                logger.debug("Finding matches...")
                matches = find_top_matches(filename, DATABASE_PATH)
                logger.debug(f"Found {len(matches)} matches")

                results = []

                bucket_name = os.getenv('GCS_BUCKET_NAME')
                gcs_prefix = os.getenv('GCS_DATABASE_PREFIX', '')

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

                    similarity = 1 - distance
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

    bucket_name = os.getenv('GCS_BUCKET_NAME')
    if bucket_name:
        logger.info(f"Using GCS bucket: {bucket_name}")
        gcs_prefix = os.getenv('GCS_DATABASE_PREFIX', '')
        if gcs_prefix:
            logger.info(f"GCS prefix: {gcs_prefix}")
    else:
        logger.warning("GCS_BUCKET_NAME not set, using local filesystem only")

    app.run(host='0.0.0.0', port=5000, debug=False)