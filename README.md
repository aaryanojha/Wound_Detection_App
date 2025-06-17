# Wound Detection Web Application

This project is a Flask-based web application that uses a machine learning model to analyze wound images. Users can upload an image of a wound and receive predictions such as wound type, confidence level, and basic first-aid advice. The application also includes an admin panel for managing datasets and training the model.

## Features

- Upload wound images for analysis
- Machine learning-based wound classification using TensorFlow/Keras
- Displays prediction confidence, urgency level, and care suggestions
- Admin panel for:
  - Uploading new datasets (ZIP format)
  - Training the model from the interface
  - Deleting old datasets
  - Login and logout functionality for access control

## Technology Stack

- Frontend: HTML, CSS, Bootstrap
- Backend: Python, Flask
- Machine Learning: TensorFlow, Keras
- File Handling: PIL, NumPy, zipfile
- Authentication: Flask sessions

## Project Structure

<pre><code>wound_detection_app/
├── static/
│ └── uploads/ # Uploaded images
├── templates/
│ ├── index.html
│ ├── result.html
│ ├── admin.html
│ └── login.html
├── dataset/ # Unzipped dataset folders
├── model/
│ └── wound_model.h5 # Trained model file
├── app.py # Main Flask app
├── utils.py # Image processing and analysis
├── train_and_save_model.py # Model training script
├── requirements.txt
└── README.md</code></pre>

bash
Copy
Edit

## How to Run Locally

1. Clone the repository:

```bash
git clone https://github.com/yourusername/wound-detection-app.git
cd wound-detection-app
Create and activate a virtual environment:

bash
Copy
Edit
python -m venv venv
source venv/bin/activate     # For Linux/macOS
venv\Scripts\activate        # For Windows
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Run the Flask app:

bash
Copy
Edit
python app.py
Then open your browser and go to http://127.0.0.1:5000.

Admin Credentials
The admin interface is protected by a login system.

Default credentials:

Username: admin

Password: admin123

These are defined in app.py and can be modified there.

Dataset Format
Upload a .zip file containing subfolders where each folder is a wound category. Each subfolder should contain relevant wound images.

Example structure:

bash
Copy
Edit
wounds_dataset.zip
├── burn/
│   ├── img1.jpg
│   ├── img2.jpg
├── cut/
│   ├── img1.jpg
│   ├── img2.jpg
The uploaded dataset is extracted into the dataset/ folder and used during model training.

Training the Model
After uploading a dataset, the admin can click the "Train Model" button to train the classifier using the uploaded data. The model will be saved to model/wound_model.h5 and used for future predictions.

Notes
Model accuracy depends on dataset quality and volume.

The app uses a simple CNN model and is meant for demonstration or learning purposes.

Predictions should not be used for real medical diagnosis or treatment.

License
This project is open-source and available to use.

Author
Aaryan Ojha
GitHub: https://github.com/aaryanojha

