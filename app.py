from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
import cv2
import numpy as np
from pathlib import Path
import base64
import os
import logging
import sys
import traceback
#from pyngrok import ngrok

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Add parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
scikit_image_dir = os.path.join(parent_dir, 'scikit-image')
sys.path.append(scikit_image_dir)

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = os.path.join(current_dir, 'static', 'uploads')
DATABASE_PATH = r"C:\Users\李晓璇\Desktop\Semester 4\CS6123\project_3\spriters\Mixed"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Create directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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
        logger.debug(f"Serving image: {filename}")
        # Convert URL path to filesystem path
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
                # Save uploaded file
                filename = os.path.join(UPLOAD_FOLDER, file.filename)
                logger.debug(f"Saving file to: {filename}")
                file.save(filename)
                
                # Import onlyedge module
                logger.debug(f"Python path: {sys.path}")
                logger.debug(f"Current directory: {os.getcwd()}")
                from Object_Recognition.onlyedge import find_top_matches
                
                # Find matches
                logger.debug("Finding matches...")
                matches = find_top_matches(filename, DATABASE_PATH)
                logger.debug(f"Found {len(matches)} matches")
                
                # Prepare results
                results = []
                for match_name, distance in matches:
                    match_path = os.path.join(DATABASE_PATH, match_name)
                    if os.path.exists(match_path):
                        # Convert distance to similarity score
                        similarity = 1 - distance
                        # Remove file extension from display name
                        display_name = os.path.splitext(match_name)[0]
                        # Convert backslashes to forward slashes for URL
                        url_path = match_path.replace('\\', '/')
                        results.append({
                            'filename': display_name,
                            'similarity': float(similarity),
                            'image_path': f'/image/{url_path}'
                        })
                
                # Get the URL for the uploaded file
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
    # Verify paths
    logger.info(f"Upload folder: {UPLOAD_FOLDER}")
    logger.info(f"Database path: {DATABASE_PATH}")
    
    if not os.path.exists(DATABASE_PATH):
        logger.error(f"Database path does not exist: {DATABASE_PATH}")
        print(f"Error: Database path does not exist: {DATABASE_PATH}")
        sys.exit(1)
    
    # # Start ngrok
    # public_url = ngrok.connect(5000)
    # print(f"\n* Public URL: {public_url}")
    # print("* Share this URL with anyone to let them access your app!")
    # print("* Press CTRL+C to quit\n")
        
    # Run app with host='0.0.0.0' to allow external access
    app.run(host='0.0.0.0', port=5000, debug=False)
