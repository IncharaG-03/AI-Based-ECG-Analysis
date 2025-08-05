import bcrypt
import matplotlib
matplotlib.use('Agg')

from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, send_from_directory
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_bcrypt import Bcrypt
from flask_mysqldb import MySQL
import MySQLdb.cursors
import numpy as np
import wfdb
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from scipy.signal import butter, filtfilt
import neurokit2 as nk
from scipy.signal import find_peaks
import os
import pandas as pd
from reportlab.lib.utils import ImageReader
import uuid
from datetime import datetime
import re
from functools import wraps
from flask import abort

app = Flask(__name__)

# Configuration
app.secret_key = "5010dfae019e413f06691431b2e3ba82bbb456c661b0d27332a4dbd5bbd36bd8"
app.config["MYSQL_HOST"] = "localhost"
app.config["MYSQL_USER"] = "root"
app.config["MYSQL_PASSWORD"] = "452003@hrX"
app.config["MYSQL_DB"] = "hospital_ecg_db"
app.config["MYSQL_CURSORCLASS"] = "DictCursor"


mysql = MySQL(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager()
login_manager.init_app(app)

DATASET_PATH = "mit-bih-arrhythmia-database-1.0.0/"
MODEL_PATH = os.path.join('model', 'model.hdf5')
CLASSES = ["Normal", "Atrial Fibrillation", "Ventricular Tachycardia", "Heart Block", "Other1", "Other2"]


try:
    model = load_model(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Check and compile if needed
if model and (not hasattr(model, 'optimizer') or model.optimizer is None):
    try:
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
    except Exception as e:
        print(f"Error compiling model: {e}")

class User(UserMixin):
    def __init__(self, id, username=None, user_type=None):
        self.id = id
        self.username = username
        self.user_type = user_type
    
    def get_type(self):
        return self.user_type

@login_manager.user_loader
def load_user(user_id):
    print(f"Loading user with ID: {user_id}")
    
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    try:
        if user_id.startswith('DR-'):
            cursor.execute("SELECT * FROM doctor WHERE Doctor_ID = %s", (user_id,))
            doctor = cursor.fetchone()
            if doctor:
                print(f"Loaded doctor user: {doctor['Doctor_ID']}")
                return User(
                    id=str(doctor["Doctor_ID"]),
                    user_type="doctor"
                )
        else:
            cursor.execute("SELECT * FROM staff WHERE Staff_Username = %s", (user_id,))
            staff = cursor.fetchone()
            if staff:
                print(f"Loaded staff user: {staff['Staff_Username']}")
                return User(
                    id=staff["Staff_Username"],
                    username=staff["Staff_Username"],
                    user_type="staff"
                )
    except MySQLdb.Error as e:
        print(f"Database error: {e}")
    finally:
        cursor.close()
    
    print("No user found")
    return None

def doctor_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if current_user.is_authenticated:
            if current_user.user_type != "doctor":
                abort(403)
        else:
            abort(401)
        return f(*args, **kwargs)
    return decorated_function

def register_doctor(username, password):
    hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
    cursor = None
    
    try:
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute("INSERT INTO doctor (Username, Password) VALUES (%s, %s)", 
                       (username, hashed_password))
        mysql.connection.commit()
        print(f"Doctor {username} registered successfully!")
        return True
    except MySQLdb.Error as e:
        print(f"Error while connecting to MySQL: {e}")
        return False
    finally:
        if cursor:
            cursor.close()

def load_ecg_sample(record_num="100"):
    try:
        record_path = f"{DATASET_PATH}{record_num}"
        record = wfdb.rdrecord(record_path)
        return record.p_signal[:, 0], record.fs
    except Exception as e:
        print(f"Error loading ECG sample: {e}")
        return None, None

def generate_ecg_plot(signal, peaks=None, title="ECG Signal"):
    try:
        unique_id = str(uuid.uuid4())[:8]
        filename = f"ecg_plot_{unique_id}.png"
        path = os.path.join("static", filename).replace("\\", "/")
        
        plt.figure(figsize=(12, 4))
        plt.plot(signal, label="ECG Signal")
        if peaks is not None:
            plt.scatter(peaks, signal[peaks], color='red', label="Peaks", marker="x")
        plt.title(title)
        plt.xlabel("Time (samples)")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.savefig(path, bbox_inches="tight", dpi=300)
        plt.close()
        return filename
    except Exception as e:
        print(f"Error generating ECG plot: {e}")
        return None

def butterworth_filter(signal, cutoff=50, fs=360, order=4):
    try:
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return filtfilt(b, a, signal)
    except Exception as e:
        print(f"Error applying Butterworth filter: {e}")
        return signal

def detect_r_peaks(ecg_signal, fs):
    try:
        processed_ecg = nk.ecg_clean(ecg_signal, sampling_rate=fs, method="pantompkins1985")
        _, r_peaks = nk.ecg_peaks(processed_ecg, sampling_rate=fs)
        return r_peaks["ECG_R_Peaks"]
    except Exception as e:
        print(f"Error detecting R-peaks: {e}")
        return np.array([])

def compute_intervals(ecg_signal, r_peaks, fs):
    try:
        rr_intervals = np.diff(r_peaks) / fs
        heart_rate = 60 / rr_intervals.mean() if len(rr_intervals) > 0 else 0
        qrs_peaks, _ = find_peaks(ecg_signal, height=np.percentile(ecg_signal, 90), distance=fs*0.06)
        qt_interval = (r_peaks[-1] - r_peaks[0]) / fs if len(r_peaks) > 1 else 0
        pr_interval = (r_peaks[1] - r_peaks[0]) / fs if len(r_peaks) > 1 else 0
        return heart_rate, qt_interval, pr_interval, qrs_peaks
    except Exception as e:
        print(f"Error computing intervals: {e}")
        return 0, 0, 0, np.array([])

def preprocess_ecg(ecg_signal):
    try:
        ecg_signal = (ecg_signal - np.mean(ecg_signal)) / np.std(ecg_signal)
        
        if len(ecg_signal) < 4096:
            padded = np.zeros(4096)
            padded[:len(ecg_signal)] = ecg_signal
            ecg_signal = padded
        elif len(ecg_signal) > 4096:
            ecg_signal = ecg_signal[:4096]
        
        if ecg_signal.ndim == 2 and ecg_signal.shape[1] == 12:
            ecg_resized = ecg_signal
        else:
            ecg_resized = np.repeat(ecg_signal[:, np.newaxis], 12, axis=1)
        
        return np.expand_dims(ecg_resized, axis=0)
    except Exception as e:
        print(f"Error preprocessing ECG: {e}")
        return np.zeros((1, 4096, 12))

def compute_framingham_risk(age, cholesterol, hdl, systolic_bp, smoker, diabetes):
    try:
        score = (0.02 * age) + (0.03 * cholesterol) - (0.05 * hdl) + (0.04 * systolic_bp) + (0.2 * smoker) + (0.15 * diabetes)
        return min(max(score, 0), 30)
    except Exception as e:
        print(f"Error computing Framingham risk: {e}")
        return 0

def compute_grace_score(age, systolic_bp, heart_rate):
    try:
        score = (0.1 * age) - (0.05 * systolic_bp) + (0.2 * heart_rate)
        return min(max(score, 0), 20)
    except Exception as e:
        print(f"Error computing GRACE score: {e}")
        return 0

def generate_pdf(predicted_class, framingham_risk, grace_score, heart_rate, qt_interval, pr_interval, ecg_filename):
    try:
        unique_id = str(uuid.uuid4())[:8]
        pdf_filename = f"ECG_Report_{unique_id}.pdf"
        pdf_path = os.path.join("static", pdf_filename).replace("\\", "/")
        
        c = canvas.Canvas(pdf_path, pagesize=letter)
        c.setFont("Helvetica-Bold", 16)
        c.drawString(100, 750, "ECG Analysis Report")
        c.setFont("Helvetica", 12)
        
        y_position = 700
        c.drawString(100, y_position, f"Prediction: {predicted_class}")
        y_position -= 30
        c.drawString(100, y_position, f"Framingham Risk: {framingham_risk:.2f}%")
        y_position -= 30
        c.drawString(100, y_position, f"GRACE Score: {grace_score:.2f}%")
        y_position -= 30
        c.drawString(100, y_position, f"Heart Rate: {heart_rate:.2f} BPM")
        
        plot_path = os.path.join("static", ecg_filename)
        if os.path.exists(plot_path):
            c.drawImage(plot_path, 100, 400, width=400, height=200)
        
        c.save()
        return pdf_filename
    except Exception as e:
        print(f"Error generating PDF: {e}")
        return None

@app.route("/", methods=["GET", "POST"])
def staff_login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        
        if not username or not password:
            flash("Please enter both username and password", "danger")
            return render_template("staff_login.html")
        
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        try:
            cursor.execute("SELECT * FROM staff WHERE Staff_Username = %s", (username,))
            staff = cursor.fetchone()
            
            if staff and bcrypt.check_password_hash(staff["Password"], password):
                staff_obj = User(
                    id=staff["Staff_Username"],
                    username=staff["Staff_Username"],
                    user_type="staff"
                )
                login_user(staff_obj)
                flash("Login successful!", "success")
                return redirect(url_for("patient_registration"))
            else:
                flash("Invalid credentials", "danger")
        except MySQLdb.Error as e:
            flash(f"Database error: {str(e)}", "danger")
        finally:
            cursor.close()
    
    return render_template("staff_login.html")

@app.route("/doctor_login", methods=["GET", "POST"])
def doctor_login():
    if request.method == "POST":
        doctor_id = request.form.get("doctor_id")
        password = request.form.get("password")
        
        if not doctor_id or not password:
            flash("Please enter both doctor ID and password", "danger")
            return render_template("doctor_login.html")
        
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        try:
            cursor.execute("SELECT * FROM doctor WHERE Doctor_ID = %s", (doctor_id,))
            doctor = cursor.fetchone()
            
            if doctor and bcrypt.check_password_hash(doctor["Password"], password):
                doctor_obj = User(
                    id=str(doctor["Doctor_ID"]),
                    user_type="doctor",
                    username=doctor['Username']  
                )
                session['doctor_name'] = doctor['Username']
                login_user(doctor_obj)
                return redirect(url_for("input_form"))
            else:
                flash("Invalid credentials", "danger")
        except MySQLdb.Error as e:
            flash(f"Database error: {str(e)}", "danger")
        finally:
            cursor.close()
    
    return render_template("doctor_login.html")

@app.route("/input_form", methods=["GET", "POST"])
@login_required
def input_form():
    doctor_name = session.get('doctor_name', 'Guest')
    patient = None
    medical_history = None
    
    if request.method == "POST":
        patient_id = request.form.get("Patient_ID")
        
        if not patient_id:
            flash("Please enter a patient ID", "danger")
            return render_template("input_form.html")
        
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        try:
            cursor.execute("SELECT * FROM patient_profile WHERE Patient_ID = %s", (patient_id,))
            patient = cursor.fetchone()
            
            if not patient:
                flash("Patient not found", "danger")
                return render_template("input_form.html")
            
            cursor.execute("""
                SELECT * FROM input 
                WHERE Patient_ID = %s 
                ORDER BY Generated_AT DESC
            """, (patient_id,))
            medical_history = cursor.fetchall()
            
        except MySQLdb.Error as e:
            flash(f"Database error: {str(e)}", "danger")
        finally:
            cursor.close()

    return render_template("input_form.html", patient=patient, medical_history=medical_history, doctor_name=doctor_name)

@app.route("/add_medical_data/<patient_id>", methods=["POST"])
@login_required
def add_medical_data(patient_id):
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    try:
        cursor.execute("SELECT * FROM patient_profile WHERE Patient_ID = %s", (patient_id,))
        patient = cursor.fetchone()
        
        if not patient:
            flash("Patient not found", "danger")
            return redirect(url_for("input_form"))
        
        systolic_bp = request.form["systolic_bp"]
        cholesterol = request.form["cholesterol"]
        hdl = request.form["hdl"]
        smoker = 1 if request.form.get("smoker") else 0
        diabetic = 1 if request.form.get("diabetic") else 0
        other_issues = request.form.get("other_issues", "")  
        generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        cursor.execute("""
    INSERT INTO input 
    (Patient_ID, Doctor_ID, Smoker, Alcoholic, Diabetic, Cholesterol, HDL, 
     Blood_Pressure, Other_Issues, Generated_AT)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
""", (
    patient_id, 
    current_user.id, 
    smoker, 
    0,  
    diabetic, 
    cholesterol, 
    hdl,
    systolic_bp, 
    "",  
    generated_at
))
        mysql.connection.commit()
        flash("Medical data saved successfully!", "success")
        
    except MySQLdb.Error as e:
        flash(f"Error saving medical data: {str(e)}", "danger")
    finally:
        cursor.close()
    
    return redirect(url_for("input_form"))

@app.route("/logout")
@login_required
def logout():
    user_type = current_user.user_type
    logout_user()
    flash("You have been logged out successfully.", "success")
    
    if user_type == "doctor":
        return redirect(url_for("doctor_login"))
    elif user_type == "staff":
        return redirect(url_for("staff_login"))
    else:
        return redirect(url_for("staff_login"))

@app.route("/automatic_analysis/<patient_id>", methods=["GET", "POST"])
@login_required
def automatic_analysis(patient_id):
    if not re.match(r'^PT-\d{5}-\d{4}$', patient_id):
        flash("Invalid Patient ID format", "danger")
        return redirect(url_for("input_form"))
    
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    try:
        cursor.execute("SELECT * FROM patient_profile WHERE Patient_ID = %s", (patient_id,))
        patient = cursor.fetchone()
    except MySQLdb.Error as e:
        flash(f"Database error: {str(e)}", "danger")
        patient = None
    finally:
        cursor.close()
    
    if not patient:
        flash("Patient not found", "danger")
        return redirect(url_for("input_form"))
    
    if request.method == "POST":
        try:
            record_num = request.form.get("record_num", "100")
            age = int(request.form.get("age", 30))
            cholesterol = int(request.form.get("cholesterol", 150))
            hdl = int(request.form.get("hdl", 40))
            systolic_bp = int(request.form.get("systolic_bp", 120))
            smoker = "smoker" in request.form
            diabetes = "diabetes" in request.form

            cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
            try:
                cursor.execute("""
                    INSERT INTO input 
                    (Patient_ID, Blood_Pressure, Cholesterol, HDL, Smoker, Diabetic, Generated_AT, Doctor_ID)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    patient_id,
                    systolic_bp,
                    cholesterol,
                    hdl,
                    smoker,
                    diabetes,
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    current_user.id
                ))
                mysql.connection.commit()
            except MySQLdb.Error as e:
                flash(f"Error saving medical data: {str(e)}", "danger")
                mysql.connection.rollback()
                return redirect(url_for("automatic_analysis", patient_id=patient_id))
            finally:
                cursor.close()

            ecg_signal, fs = load_ecg_sample(record_num)
            if ecg_signal is None:
                flash("Failed to load ECG sample", "danger")
                return redirect(url_for("automatic_analysis", patient_id=patient_id))
            
            ecg_signal = butterworth_filter(ecg_signal, fs=fs)
            r_peaks = detect_r_peaks(ecg_signal, fs)
            heart_rate, qt_interval, pr_interval, _ = compute_intervals(ecg_signal, r_peaks, fs)

            ecg_filename = generate_ecg_plot(ecg_signal, r_peaks, "ECG with Anomalies")
            if not ecg_filename:
                flash("Failed to generate ECG plot", "danger")
                return redirect(url_for("automatic_analysis", patient_id=patient_id))
            
            if model is None:
                flash("Model not available for prediction", "danger")
                return redirect(url_for("automatic_analysis", patient_id=patient_id))
            
            prediction = model.predict(preprocess_ecg(ecg_signal))
            predicted_class = CLASSES[np.argmax(prediction)]
            
            pdf_filename = generate_pdf(
                predicted_class,
                compute_framingham_risk(age, cholesterol, hdl, systolic_bp, smoker, diabetes),
                compute_grace_score(age, systolic_bp, heart_rate),
                heart_rate,
                qt_interval,
                pr_interval,
                ecg_filename
            )

            return render_template("result.html",
                predicted_class=predicted_class,
                framingham_risk=compute_framingham_risk(age, cholesterol, hdl, systolic_bp, smoker, diabetes),
                grace_score=compute_grace_score(age, systolic_bp, heart_rate),
                heart_rate=heart_rate,
                qt_interval=qt_interval,
                pr_interval=pr_interval,
                ecg_filename=ecg_filename,
                pdf_filename=pdf_filename,
                patient=patient
            )

        except Exception as e:
            app.logger.error(f"Automatic analysis error: {str(e)}")
            flash(f"Analysis failed: {str(e)}", "danger")
            return redirect(url_for("automatic_analysis", patient_id=patient_id))

    return render_template("automatic_analysis.html", patient=patient)

@app.route("/download_report/<filename>")
@login_required
def download_report(filename):
    return send_from_directory(
        directory=os.path.join(app.root_path, 'static'),
        path=filename,
        as_attachment=True
    )

@app.route("/dashboard")
@doctor_required
def dashboard():
    return render_template("dashboard.html", username=current_user.username)

def generate_patient_id():
    current_year = datetime.now().year
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    
    try:
        cursor.execute("""
            SELECT Patient_ID 
            FROM patient_profile 
            WHERE Patient_ID LIKE CONCAT('PT-%%', %s) 
            ORDER BY Patient_ID DESC 
            LIMIT 1
        """, (current_year,))
        
        last_patient = cursor.fetchone()
        if last_patient:
            last_num = int(last_patient['Patient_ID'].split('-')[1])
            new_num = last_num + 1
        else:
            new_num = 10001
        
        new_patient_id = f"PT-{new_num}-{current_year}"
        return new_patient_id
    except MySQLdb.Error as e:
        app.logger.error(f"Database error generating patient ID: {str(e)}")
        raise
    finally:
        cursor.close()

def validate_doctor_id(doctor_id):
    pattern = r'^DR-\d{3}-\d{4}$'
    return re.match(pattern, doctor_id) is not None

def validate_email(email):
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

@app.route("/patient_registration", methods=["GET", "POST"])
@login_required
def patient_registration():
    patient = None
    errors = {}
    form_data = {}
    
    if request.method == "POST":
        doctor_id = request.form.get("Doctor_ID", "")
        patient_name = request.form.get("Patient_Name", "")
        age = request.form.get("Age", "")
        gender = request.form.get("Gender", "")
        address = request.form.get("Address", "")
        email_id = request.form.get("Email_ID", "")
        personal_contact = request.form.get("Personal_Contact", "")
        emergency_contact = request.form.get("Emergency_Contact", "")
        
        form_data = {
            "Doctor_ID": doctor_id,
            "Patient_Name": patient_name,
            "Age": age,
            "Gender": gender,
            "Address": address,
            "Email_ID": email_id,
            "Personal_Contact": personal_contact,
            "Emergency_Contact": emergency_contact
        }
        
        if not patient_name.strip():
            errors["Patient_Name"] = "Please enter the patient's full name."
        
        try:
            age = int(age)
            if age < 1 or age > 150:
                errors["Age"] = "Please enter a valid age between 1 and 150."
        except (ValueError, TypeError):
            errors["Age"] = "Please enter a valid age."
        
        if not gender:
            errors["Gender"] = "Please select a gender."
        
        if not address.strip():
            errors["Address"] = "Please enter the patient's address."
        
        if not email_id or not validate_email(email_id):
            errors["Email_ID"] = "Please enter a valid email address."
        
        if not personal_contact or not personal_contact.isdigit() or len(personal_contact) != 10:
            errors["Personal_Contact"] = "Please enter a valid 10-digit phone number."
        
        if not emergency_contact or not emergency_contact.isdigit() or len(emergency_contact) != 10:
            errors["Emergency_Contact"] = "Please enter a valid 10-digit phone number."
        
        if personal_contact == emergency_contact:
            errors["Emergency_Contact"] = "Personal contact and emergency contact cannot be the same."
        
        if not doctor_id or not validate_doctor_id(doctor_id):
            errors["Doctor_ID"] = "Please enter a valid doctor ID in the format DR-001-2024."
        
        if not errors.get("Email_ID"):
            try:
                cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
                cursor.execute("SELECT * FROM patient_profile WHERE Email_ID = %s", (email_id,))
                existing_email = cursor.fetchone()
                if existing_email:
                    errors["Email_ID"] = "Email ID already exists. Please use a different email address."
                cursor.close()
            except MySQLdb.Error as e:
                flash(f"Database error: {str(e)}", "danger")
        
        if errors:
            return render_template("patient_registration.html", errors=errors, form_data=form_data, patient=patient)
        
        try:
            patient_id = generate_patient_id()

            cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
            cursor.execute(
                """INSERT INTO patient_profile 
                   (Patient_ID, Patient_Name, Age, Gender, Address, Email_ID, Personal_Contact, 
                    Emergency_Contact, Doctor_ID, Created_At, Staff_Username)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (patient_id, patient_name, age, gender, address, email_id, personal_contact,
                 emergency_contact, doctor_id, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), current_user.username)
            )
            mysql.connection.commit()
            flash("Patient registered successfully!", "success")
            
            cursor.execute("SELECT * FROM patient_profile WHERE Patient_ID = %s", (patient_id,))
            patient = cursor.fetchone()
            cursor.close()
            
        except MySQLdb.Error as e:
            flash(f"Error registering patient: {str(e)}", "danger")

    return render_template("patient_registration.html", patient=patient, errors=errors, form_data=form_data)

@app.route("/edit_patient/<patient_id>", methods=["GET", "POST"])
@login_required
def edit_patient(patient_id):
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    try:
        if request.method == "POST":
            patient_name = request.form.get("Patient_Name", "")
            age = request.form.get("Age", "")
            gender = request.form.get("Gender", "")
            address = request.form.get("Address", "")
            email_id = request.form.get("Email_ID", "")
            personal_contact = request.form.get("Personal_Contact", "")
            emergency_contact = request.form.get("Emergency_Contact", "")
            doctor_id = request.form.get("Doctor_ID", "")

            cursor.execute(
                """UPDATE patient_profile 
                   SET Patient_Name = %s, Age = %s, Gender = %s, Address = %s, Email_ID = %s, 
                       Personal_Contact = %s, Emergency_Contact = %s, Doctor_ID = %s
                   WHERE Patient_ID = %s""",
                (patient_name, age, gender, address, email_id, personal_contact,
                 emergency_contact, doctor_id, patient_id)
            )
            mysql.connection.commit()
            flash("Patient details updated successfully!", "success")
            return redirect(url_for("patient_registration"))
        
        cursor.execute("SELECT * FROM patient_profile WHERE Patient_ID = %s", (patient_id,))
        patient = cursor.fetchone()
        if not patient:
            flash("Patient not found", "danger")
            return redirect(url_for("patient_registration"))
        
        return render_template("edit_patient.html", patient=patient)
    except MySQLdb.Error as e:
        flash(f"Error updating patient details: {str(e)}", "danger")
    finally:
        cursor.close()
    
    return redirect(url_for("patient_registration"))

@app.route("/debug")
def debug():
    return jsonify({
        "user_id": session.get("_user_id"),
        "is_authenticated": current_user.is_authenticated,
        "user_type": current_user.user_type if current_user.is_authenticated else None
    })

if __name__ == "__main__":
    os.makedirs("static", exist_ok=True)
    app.run(debug=True)




# @app.route("/manual_analysis", methods=["GET", "POST"])
# @login_required
# def manual_analysis():
#     if request.method == "POST":
#         try:
#             # Process form data
#             params = {k: float(request.form[k]) for k in ['p_peak', 'qrs_interval', 'qt_interval', 'pr_interval', 'heart_rate']}
#             demographics = {k: int(request.form[k]) for k in ['age', 'cholesterol', 'hdl', 'systolic_bp']}
            
#             # Generate ECG signal
#             ecg_signal = simulate_ecg_signal(**params, fs=360)
            
#             # Generate plot
#             ecg_filename = generate_ecg_plot(ecg_signal[:, 0], title="Simulated ECG")
            
#             # Get model prediction
#             prediction = model.predict(preprocess_ecg(ecg_signal))
#             predicted_class = CLASSES[np.argmax(prediction)]
            
#             # Generate PDF
#             pdf_filename = generate_pdf(
#                 predicted_class,
#                 compute_framingham_risk(**demographics, smoker="smoker" in request.form, diabetes="diabetes" in request.form),
#                 compute_grace_score(demographics['age'], demographics['systolic_bp'], params['heart_rate']),
#                 params['heart_rate'],
#                 params['qt_interval'],
#                 params['pr_interval'],
#                 ecg_filename
#             )

#             return render_template("result.html",
#                 predicted_class=predicted_class,
#                 framingham_risk=compute_framingham_risk(**demographics, smoker="smoker" in request.form, diabetes="diabetes" in request.form),
#                 grace_score=compute_grace_score(demographics['age'], demographics['systolic_bp'], params['heart_rate']),
#                 heart_rate=params['heart_rate'],
#                 qt_interval=params['qt_interval'],
#                 pr_interval=params['pr_interval'],
#                 ecg_filename=ecg_filename,
#                 pdf_filename=pdf_filename
#             )

#         except Exception as e:
#             app.logger.error(f"Manual analysis error: {str(e)}")
#             flash(f"Analysis failed: {str(e)}", "danger")
#             return redirect(url_for("manual_analysis"))

#     return render_template("manual_analysis.html")
