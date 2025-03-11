import os
import shutil
import tempfile
import zipfile
import pandas as pd
import numpy as np
import sqlite3
from flask import Flask, request, jsonify, render_template, send_file
from flask_sqlalchemy import SQLAlchemy
from ultralytics import YOLO
from easyocr import Reader
from fuzzywuzzy import fuzz, process
import re
import cv2

app = Flask(__name__)

# Directories for file uploads and results
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['RESULTS_FOLDER'] = 'results/'

# Temporary directory for intermediate file operations
TEMP_DIR = tempfile.gettempdir()

# Initialize YOLO models and EasyOCR reader
classifier = YOLO(r"C:\Users\Admin\Downloads\AI-based-fraud-management-system-for-UID-Aadhar-main 2\AI-based-fraud-management-system-for-UID-Aadhar-main\classification\runs\classify\train\weights\best.pt")
detector = YOLO(r"C:\Users\Admin\Downloads\AI-based-fraud-management-system-for-UID-Aadhar-main 2\AI-based-fraud-management-system-for-UID-Aadhar-main\detection\runs\detect\train\weights\best.pt")
reader = Reader(['en'])

# SQLite database initialization
DATABASE_PATH = 'users.db'

def init_db():
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            SrNo TEXT PRIMARY KEY NOT NULL,
            name_match_score INTEGER,
            uid_match_score INTEGER,
            final_address_match_score INTEGER,
            overall_score INTEGER,
            final_remarks TEXT,
            document_type TEXT,
            accepted_rejected TEXT
        )
    ''')
    conn.commit()
    conn.close()

# Initialize database on startup
init_db()

# Helper functions
def process_image(image_path):
    if classifier.predict(image_path)[0].probs.numpy().top1 == 0:
        fields = detector(image_path)
        image = cv2.imread(image_path)
        extracted_data = {}
        for field in fields[0].boxes.data.tolist():
            x1, y1, x2, y2, confidence, class_id = map(int, field[:6])
            field_class = detector.names[class_id]
            cropped_roi = image[y1:y2, x1:x2]
            gray_roi = cv2.cvtColor(cropped_roi, cv2.COLOR_BGR2GRAY)
            text = reader.readtext(gray_roi, detail=0)
            extracted_data[field_class] = ' '.join(text)
        return extracted_data
    return None

def normalize_text(text):
    if not text:
        return "text empty"
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.split()).lower()

def calculate_match_score(input_value, extracted_value):
    if pd.isna(input_value) or pd.isna(extracted_value):
        return 0
    return fuzz.ratio(str(input_value), str(extracted_value))

def compare_data(input_data, json_data):
    excel_data = input_data.copy()
    overall_score = 0
    for idx, row in excel_data.iterrows():
        serial_no = row.get("SrNo")
        uid = row.get("UID")
        extracted = json_data.get(serial_no)

        if extracted:
            extracted_uid = extracted.get("uid", "").replace(" ", "")
            extracted_name = extracted.get("name", "")
            extracted_address = extracted.get("address", "")
            row['Extracted UID'] = extracted_uid
            row['Extracted Name'] = extracted_name
            row['Extracted Address'] = extracted_address
            # UID Match
            uid_match = uid == extracted_uid
            uid_score = 100 if uid_match else 0
            row['UID Match Score'] = uid_score

            # Name Match
            name_score = calculate_match_score(row.get("Name"), extracted_name)
            row['Name Match Score'] = name_score
            row['Name Match Percentage'] = name_score

            # Address Match
            address_score = fuzz.token_set_ratio(normalize_text(row.get("Address", "")), normalize_text(extracted_address))
            row['Address Match Score'] = address_score
            row['Overall Score'] = (uid_score + name_score + address_score) / 3
            row['Final Remarks'] = "All matched" if uid_match and name_score > 80 and address_score > 80 else "Mismatch"

        excel_data.loc[idx] = row
    return excel_data

# Flask route for uploading files
@app.route('/upload', methods=['POST'])
def upload_files():
    removeDb()
    delete_uploads("uploads/")  # Clean previous uploads
    init_db()  # Reinitialize the database

    # Check for required files in the request
    if 'zipfile' in request.files and 'excelfile' in request.files:
        zip_file = request.files['zipfile']
        excel_file = request.files['excelfile']

        # Save files to the upload folder
        zip_path = os.path.join(app.config['UPLOAD_FOLDER'], zip_file.filename)
        excel_path = os.path.join(app.config['UPLOAD_FOLDER'], excel_file.filename)
        zip_file.save(zip_path)
        excel_file.save(excel_path)

        # Unzip and process images from the ZIP file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(app.config['UPLOAD_FOLDER'])

        image_paths = [os.path.join(app.config['UPLOAD_FOLDER'], f) for f in os.listdir(app.config['UPLOAD_FOLDER']) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        processed_results = {}

        for image_path in image_paths:
            file_name = os.path.basename(image_path)
            key = file_name.split('.')[0][:3]
            if key not in processed_results:
                extracted_data = process_image(image_path)
                if extracted_data:
                    processed_results[key] = extracted_data

        # Read the Excel file and compare data
        df = pd.read_excel(excel_path)
        df = df.astype('str')
        comparison_results = compare_data(df, processed_results)

        # Save results to a new Excel file
        results_df = pd.DataFrame(comparison_results)
        os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
        results_file_path = os.path.join(app.config['RESULTS_FOLDER'], 'results.xlsx')
        results_df.to_excel(results_file_path, index=False)

        return jsonify({"message": "Files processed successfully!", 
                        "results": comparison_results[['Name Match Score', 'UID Match Score', 'SrNo', 'Final Address Match Score', 'Overall Score', 'Final Remarks']].to_dict(orient='records')})

    return jsonify({"error": "Both ZIP and Excel files are required."}), 400

# Flask route to download the results
@app.route('/download', methods=['GET'])
def download_results():
    file_path = os.path.join(app.config['RESULTS_FOLDER'], 'results.xlsx')
    return send_file(file_path, as_attachment=True)

# Flask route to clear the uploads folder
def delete_uploads(UPLOAD_FOLDER):
    try:
        if os.path.exists(UPLOAD_FOLDER):
            for filename in os.listdir(UPLOAD_FOLDER):
                file_path = os.path.join(UPLOAD_FOLDER, filename)
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            return jsonify({"message": "Uploads folder contents deleted successfully."}), 200
        else:
            return jsonify({"message": "Uploads folder does not exist."}), 404
    except Exception as e:
        return jsonify({"error": f"Failed to delete uploads folder contents: {str(e)}"}), 500

# Flask route to delete the database
def removeDb():
    if os.path.exists(DATABASE_PATH):
        try:
            os.remove(DATABASE_PATH)
            return jsonify({"message": "Database file deleted successfully."}), 200
        except Exception as e:
            return jsonify({"error": f"Failed to delete database file: {str(e)}"}), 500
    else:
        return jsonify({"message": "Database file does not exist."}), 404

# Route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
