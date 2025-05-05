import os
import cv2
import numpy as np
import csv
import time
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Configure folders
UPLOAD_FOLDER = 'static/dataset'
TRAINER_FOLDER = 'static/trainer'
ATTENDANCE_FOLDER = 'static/Attendance'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['TRAINER_FOLDER'] = TRAINER_FOLDER
app.config['ATTENDANCE_FOLDER'] = ATTENDANCE_FOLDER

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TRAINER_FOLDER, exist_ok=True)
os.makedirs(ATTENDANCE_FOLDER, exist_ok=True)

# Initialize face detector and recognizer
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        # Get form data
        user_id = request.form['user_id']
        name = request.form['name']
        
        # Initialize webcam
        cam = cv2.VideoCapture(0)
        cam.set(3, 640)  # width
        cam.set(4, 480)  # height
        
        # Create directory for user
        user_folder = os.path.join(app.config['UPLOAD_FOLDER'], user_id)
        os.makedirs(user_folder, exist_ok=True)
        
        # Capture 100 face samples
        count = 0
        while count < 100:
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                count += 1
                
                # Save the captured image
                cv2.imwrite(f"{user_folder}/{user_id}_{name}_{count}.jpg", gray[y:y+h, x:x+w])
                cv2.imshow('Register Face', img)
            
            k = cv2.waitKey(100) & 0xff
            if k == 27:  # ESC key
                break
        
        cam.release()
        cv2.destroyAllWindows()
        
        flash(f'Successfully registered {name} with ID {user_id}')
        return redirect(url_for('home'))
    
    return render_template('register.html')

@app.route('/train', methods=['GET', 'POST'])
def train():
    if request.method == 'POST':
        # Get the training images and labels
        faces = []
        ids = []
        
        for root, dirs, files in os.walk(app.config['UPLOAD_FOLDER']):
            for file in files:
                if file.endswith("jpg") or file.endswith("jpeg") or file.endswith("png"):
                    path = os.path.join(root, file)
                    user_id = int(os.path.basename(root))
                    
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    face = face_detector.detectMultiScale(img)
                    
                    for (x, y, w, h) in face:
                        faces.append(img[y:y+h, x:x+w])
                        ids.append(user_id)
        
        # Train the model
        recognizer.train(faces, np.array(ids))
        recognizer.save(os.path.join(app.config['TRAINER_FOLDER'], 'trainer.yml'))
        
        flash('Model trained successfully!')
        return redirect(url_for('home'))
    
    return render_template('train.html')

@app.route('/recognize', methods=['GET', 'POST'])
def recognize():
    if request.method == 'POST':
        # Load the trained model
        recognizer.read(os.path.join(app.config['TRAINER_FOLDER'], 'trainer.yml'))
        
        # Initialize webcam
        cam = cv2.VideoCapture(0)
        cam.set(3, 640)  # width
        cam.set(4, 480)  # height
        
        # Load names from dataset
        names = {}
        for root, dirs, files in os.walk(app.config['UPLOAD_FOLDER']):
            for file in files:
                if file.endswith("jpg") or file.endswith("jpeg") or file.endswith("png"):
                    user_id = int(os.path.basename(root))
                    name = file.split('_')[1]
                    names[user_id] = name
        
        # Create attendance file for today
        today = datetime.now().strftime("%Y-%m-%d")
        attendance_file = os.path.join(app.config['ATTENDANCE_FOLDER'], f'attendance_{today}.csv')
        
        # Check if file exists, if not create with header
        if not os.path.exists(attendance_file):
            with open(attendance_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['ID', 'Name', 'Time'])
        
        # Read existing attendance to avoid duplicates
        attended_ids = set()
        if os.path.exists(attendance_file):
            with open(attendance_file, 'r') as f:
                reader = csv.reader(f)
                next(reader)  # skip header
                for row in reader:
                    if row:  # check if row is not empty
                        attended_ids.add(int(row[0]))
        
        # Face recognition loop
        recognized = False
        start_time = time.time()
        while time.time() - start_time < 10:  # Run for 10 seconds
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
                
                # Check if confidence is less than 50 (lower is more confident)
                if confidence < 50 and id in names:
                    name = names[id]
                    confidence_text = f"{round(100 - confidence, 2)}%"
                    
                    cv2.putText(img, name, (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.putText(img, confidence_text, (x+5, y+h-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1)
                    
                    # Mark attendance if not already done
                    if id not in attended_ids:
                        attended_ids.add(id)
                        with open(attendance_file, 'a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([id, name, datetime.now().strftime("%H:%M:%S")])
                        recognized = True
                        flash(f'Attendance marked for {name}')
                else:
                    name = "Unknown"
                    cv2.putText(img, name, (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            cv2.imshow('Recognize Face', img)
            k = cv2.waitKey(10) & 0xff
            if k == 27 or recognized:  # ESC key or recognized
                break
        
        cam.release()
        cv2.destroyAllWindows()
        
        if not recognized:
            flash('No recognized faces or attendance already marked')
        return redirect(url_for('home'))
    
    return render_template('recognize.html')

@app.route('/attendance')
def attendance():
    # Get all attendance files
    attendance_files = []
    for file in os.listdir(app.config['ATTENDANCE_FOLDER']):
        if file.startswith('attendance_') and file.endswith('.csv'):
            date = file[11:-4]  # Extract date from filename
            attendance_files.append((date, file))
    
    # Sort by date (newest first)
    attendance_files.sort(reverse=True)
    
    # Get specific file if requested
    date = request.args.get('date')
    records = []
    if date:
        file_path = os.path.join(app.config['ATTENDANCE_FOLDER'], f'attendance_{date}.csv')
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                reader = csv.reader(f)
                next(reader)  # skip header
                records = list(reader)
    
    return render_template('attendance.html', attendance_files=attendance_files, records=records, selected_date=date)

if __name__ == '__main__':
    app.run(debug=True)

