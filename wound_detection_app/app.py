# app.py
from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.utils import secure_filename
import os
import threading
import zipfile
import shutil
from utils import analyze_wound
import json

STATUS_FILE = 'training_status.json'

app = Flask(__name__)
app.secret_key = "your_secret_key"  # Required for flashing messages

ADMIN_USERNAME = 'admin'
ADMIN_PASSWORD = 'admin123'

UPLOAD_FOLDER = 'static/uploads'
DATASET_FOLDER = 'dataset'
MODEL_FOLDER = 'model'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATASET_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/admin-login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session['admin_logged_in'] = True
            flash('Login successful! 👋', 'success')
            return redirect(url_for('admin'))
        else:
            flash('Invalid credentials ❌', 'danger')
    return render_template('admin_login.html')

@app.route('/admin')
def admin():
    if not session.get('admin_logged_in'):
        flash('Please log in first.', 'warning')
        return redirect(url_for('admin_login'))
        
    uploaded_file = request.args.get('file')
    return render_template('admin.html', uploaded_file=uploaded_file)

@app.route('/admin-logout')
def admin_logout():
    session.pop('admin_logged_in', None)
    flash('Logged out successfully! 👋', 'info')
    return redirect(url_for('index'))

@app.route('/upload-dataset', methods=['POST'])
def upload_dataset():
    file = request.files.get('dataset')
    if file and file.filename.endswith('.zip'):
        filename = secure_filename(file.filename)
        filepath = os.path.join('temp_uploads', filename)
        os.makedirs('temp_uploads', exist_ok=True)
        file.save(filepath)

        # Extract and overwrite dataset
        if os.path.exists('dataset'):
            shutil.rmtree('dataset')
        os.makedirs('dataset', exist_ok=True)
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall('dataset')

        os.remove(filepath)
        flash(f"✅ Dataset '{filename}' uploaded and extracted successfully!", 'success')
        return redirect(url_for('admin', file=filename))
    else:
        flash("❌ Invalid file. Please upload a ZIP file.", 'error')
        return redirect(url_for('admin'))
    
@app.route('/delete-dataset', methods=['POST'])
def delete_dataset():
    try:
        if os.path.exists(DATASET_FOLDER):
            shutil.rmtree(DATASET_FOLDER)
            os.makedirs(DATASET_FOLDER, exist_ok=True)
            flash("✅ Dataset deleted successfully!", "success")
        else:
            flash("⚠️ Dataset folder not found.", "warning")
    except Exception as e:
        flash(f"❌ Error deleting dataset: {str(e)}", "danger")
    
    return redirect(url_for('admin'))

@app.route('/analyze', methods=['POST'])
def analyze():
    image = request.files['image']
    if not image:
        return 'No image uploaded', 400

    filename = secure_filename(image.filename)
    image_path = os.path.join(UPLOAD_FOLDER, filename)
    image.save(image_path)

    predicted_class, confidence, result = analyze_wound(image_path)

    # Build the relative URL to the image so the frontend can display it
    image_url = url_for('static', filename=f'uploads/{filename}')

    return render_template('result.html', result=result, image_url=image_url)

@app.route('/train', methods=['POST'])
def train_model():
    def background_train():
        try:
            from train_and_save_model import train_and_save
            train_and_save()
            with open(STATUS_FILE, 'w') as f:
                json.dump({"status": "completed"}, f)
        except Exception as e:
            with open(STATUS_FILE, 'w') as f:
                json.dump({"status": f"failed: {str(e)}"}, f)

    # Set initial status
    with open(STATUS_FILE, 'w') as f:
        json.dump({"status": "in_progress"}, f)

    thread = threading.Thread(target=background_train)
    thread.start()
    return redirect(url_for('admin'))

@app.route('/training-status')
def training_status():
    try:
        with open(STATUS_FILE) as f:
            return json.load(f)
    except:
        return {"status": "unknown"}


if __name__ == '__main__':
    app.run(debug=True)
