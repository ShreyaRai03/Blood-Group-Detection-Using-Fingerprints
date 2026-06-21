"""
Blood Group Detection from Fingerprint using Deep Learning
Complete implementation with GUI, login system, and report generation
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
from PIL import Image, ImageTk
import cv2
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import hashlib

# ==================== MODEL ARCHITECTURE ====================

def create_fingerprint_feature_extractor():
    """Create CNN for fingerprint feature extraction"""
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu')
    ])
    return model

def create_blood_group_classifier():
    """Create classifier for blood group prediction"""
    model = models.Sequential([
        layers.Input(shape=(128,)),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dense(8, activation='softmax')  # 8 blood groups: A+, A-, B+, B-, AB+, AB-, O+, O-
    ])
    return model

def create_full_model():
    """Combine feature extractor and classifier"""
    feature_extractor = create_fingerprint_feature_extractor()
    classifier = create_blood_group_classifier()
    
    model = models.Sequential([
        feature_extractor,
        classifier
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# ==================== DATA PROCESSING ====================

def preprocess_fingerprint(image_path):
    """Preprocess fingerprint image"""
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        
        # Resize to standard size
        img = cv2.resize(img, (128, 128))
        
        # Apply histogram equalization
        img = cv2.equalizeHist(img)
        
        # Normalize
        img = img.astype('float32') / 255.0
        
        # Add batch and channel dimensions
        img = np.expand_dims(img, axis=-1)
        img = np.expand_dims(img, axis=0)
        
        return img
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

# ==================== USER DATABASE ====================

class UserDatabase:
    """Simple JSON-based user database"""
    
    def __init__(self, db_file='users.json'):
        self.db_file = db_file
        self.users = self.load_database()
    
    def load_database(self):
        """Load users from JSON file"""
        if os.path.exists(self.db_file):
            with open(self.db_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_database(self):
        """Save users to JSON file"""
        with open(self.db_file, 'w') as f:
            json.dump(self.users, f, indent=4)
    
    def hash_password(self, password):
        """Hash password using SHA256"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def register_user(self, username, password, email):
        """Register new user"""
        if username in self.users:
            return False, "Username already exists"
        
        self.users[username] = {
            'password': self.hash_password(password),
            'email': email,
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'detections': []
        }
        self.save_database()
        return True, "User registered successfully"
    
    def authenticate(self, username, password):
        """Authenticate user"""
        if username not in self.users:
            return False, "User not found"
        
        if self.users[username]['password'] == self.hash_password(password):
            return True, "Login successful"
        return False, "Invalid password"
    
    def add_detection(self, username, detection_data):
        """Add detection result to user history"""
        if username in self.users:
            self.users[username]['detections'].append(detection_data)
            self.save_database()

# ==================== GUI APPLICATION ====================

class BloodGroupDetectorApp:
    """Main application class"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Blood Group Detection from Fingerprint")
        self.root.geometry("900x700")
        self.root.configure(bg='#f0f0f0')
        
        self.user_db = UserDatabase()
        self.current_user = None
        self.model = None
        self.blood_groups = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']
        
        # Initialize or load model
        self.initialize_model()
        
        # Show login screen
        self.show_login_screen()
    
    def initialize_model(self):
        """Initialize or load the model"""
        model_path = 'blood_group_model.h5'
        
        if os.path.exists(model_path):
            try:
                self.model = keras.models.load_model(model_path)
                print("Model loaded successfully")
            except:
                print("Creating new model")
                self.model = create_full_model()
        else:
            print("Creating new model")
            self.model = create_full_model()
            # For demonstration, we'll use a randomly initialized model
            # In production, you would train this on actual data
    
    def clear_window(self):
        """Clear all widgets from window"""
        for widget in self.root.winfo_children():
            widget.destroy()
    
    # ==================== LOGIN SCREEN ====================
    
    def show_login_screen(self):
        """Display login screen"""
        self.clear_window()
        
        # Header
        header = tk.Frame(self.root, bg='#2c3e50', height=80)
        header.pack(fill='x')
        
        title = tk.Label(header, text="🔬 Blood Group Detection System", 
                        font=('Arial', 24, 'bold'), fg='white', bg='#2c3e50')
        title.pack(pady=20)
        
        # Main frame
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(expand=True, fill='both', padx=50, pady=50)
        
        # Login container
        login_frame = tk.LabelFrame(main_frame, text="Login", font=('Arial', 14, 'bold'),
                                    bg='white', padx=30, pady=30)
        login_frame.pack(expand=True)
        
        # Username
        tk.Label(login_frame, text="Username:", font=('Arial', 12), bg='white').grid(row=0, column=0, sticky='w', pady=10)
        self.username_entry = tk.Entry(login_frame, font=('Arial', 12), width=25)
        self.username_entry.grid(row=0, column=1, pady=10, padx=10)
        
        # Password
        tk.Label(login_frame, text="Password:", font=('Arial', 12), bg='white').grid(row=1, column=0, sticky='w', pady=10)
        self.password_entry = tk.Entry(login_frame, font=('Arial', 12), width=25, show='*')
        self.password_entry.grid(row=1, column=1, pady=10, padx=10)
        
        # Buttons
        btn_frame = tk.Frame(login_frame, bg='white')
        btn_frame.grid(row=2, column=0, columnspan=2, pady=20)
        
        tk.Button(btn_frame, text="Login", command=self.login, 
                 font=('Arial', 12, 'bold'), bg='#27ae60', fg='white',
                 padx=30, pady=10, cursor='hand2').pack(side='left', padx=5)
        
        tk.Button(btn_frame, text="Register", command=self.show_register_screen,
                 font=('Arial', 12), bg='#3498db', fg='white',
                 padx=30, pady=10, cursor='hand2').pack(side='left', padx=5)
    
    def show_register_screen(self):
        """Display registration screen"""
        self.clear_window()
        
        # Header
        header = tk.Frame(self.root, bg='#2c3e50', height=80)
        header.pack(fill='x')
        
        title = tk.Label(header, text="🔬 User Registration", 
                        font=('Arial', 24, 'bold'), fg='white', bg='#2c3e50')
        title.pack(pady=20)
        
        # Main frame
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(expand=True, fill='both', padx=50, pady=50)
        
        # Register container
        register_frame = tk.LabelFrame(main_frame, text="Create Account", font=('Arial', 14, 'bold'),
                                       bg='white', padx=30, pady=30)
        register_frame.pack(expand=True)
        
        # Username
        tk.Label(register_frame, text="Username:", font=('Arial', 12), bg='white').grid(row=0, column=0, sticky='w', pady=10)
        self.reg_username_entry = tk.Entry(register_frame, font=('Arial', 12), width=25)
        self.reg_username_entry.grid(row=0, column=1, pady=10, padx=10)
        
        # Email
        tk.Label(register_frame, text="Email:", font=('Arial', 12), bg='white').grid(row=1, column=0, sticky='w', pady=10)
        self.reg_email_entry = tk.Entry(register_frame, font=('Arial', 12), width=25)
        self.reg_email_entry.grid(row=1, column=1, pady=10, padx=10)
        
        # Password
        tk.Label(register_frame, text="Password:", font=('Arial', 12), bg='white').grid(row=2, column=0, sticky='w', pady=10)
        self.reg_password_entry = tk.Entry(register_frame, font=('Arial', 12), width=25, show='*')
        self.reg_password_entry.grid(row=2, column=1, pady=10, padx=10)
        
        # Confirm Password
        tk.Label(register_frame, text="Confirm Password:", font=('Arial', 12), bg='white').grid(row=3, column=0, sticky='w', pady=10)
        self.reg_confirm_entry = tk.Entry(register_frame, font=('Arial', 12), width=25, show='*')
        self.reg_confirm_entry.grid(row=3, column=1, pady=10, padx=10)
        
        # Buttons
        btn_frame = tk.Frame(register_frame, bg='white')
        btn_frame.grid(row=4, column=0, columnspan=2, pady=20)
        
        tk.Button(btn_frame, text="Register", command=self.register,
                 font=('Arial', 12, 'bold'), bg='#27ae60', fg='white',
                 padx=30, pady=10, cursor='hand2').pack(side='left', padx=5)
        
        tk.Button(btn_frame, text="Back to Login", command=self.show_login_screen,
                 font=('Arial', 12), bg='#95a5a6', fg='white',
                 padx=30, pady=10, cursor='hand2').pack(side='left', padx=5)
    
    def login(self):
        """Handle login"""
        username = self.username_entry.get()
        password = self.password_entry.get()
        
        if not username or not password:
            messagebox.showerror("Error", "Please fill in all fields")
            return
        
        success, message = self.user_db.authenticate(username, password)
        
        if success:
            self.current_user = username
            messagebox.showinfo("Success", message)
            self.show_main_screen()
        else:
            messagebox.showerror("Error", message)
    
    def register(self):
        """Handle registration"""
        username = self.reg_username_entry.get()
        email = self.reg_email_entry.get()
        password = self.reg_password_entry.get()
        confirm = self.reg_confirm_entry.get()
        
        if not all([username, email, password, confirm]):
            messagebox.showerror("Error", "Please fill in all fields")
            return
        
        if password != confirm:
            messagebox.showerror("Error", "Passwords do not match")
            return
        
        if len(password) < 6:
            messagebox.showerror("Error", "Password must be at least 6 characters")
            return
        
        success, message = self.user_db.register_user(username, password, email)
        
        if success:
            messagebox.showinfo("Success", message)
            self.show_login_screen()
        else:
            messagebox.showerror("Error", message)
    
    # ==================== MAIN SCREEN ====================
    
    def show_main_screen(self):
        """Display main detection screen"""
        self.clear_window()
        
        # Header
        header = tk.Frame(self.root, bg='#2c3e50', height=60)
        header.pack(fill='x')
        
        title = tk.Label(header, text=f"🔬 Blood Group Detection - Welcome {self.current_user}", 
                        font=('Arial', 18, 'bold'), fg='white', bg='#2c3e50')
        title.pack(side='left', padx=20, pady=15)
        
        logout_btn = tk.Button(header, text="Logout", command=self.logout,
                              font=('Arial', 10), bg='#e74c3c', fg='white',
                              padx=15, pady=5, cursor='hand2')
        logout_btn.pack(side='right', padx=20)
        
        # Main container
        container = tk.Frame(self.root, bg='#f0f0f0')
        container.pack(expand=True, fill='both', padx=20, pady=20)
        
        # Left panel - Image upload and preview
        left_panel = tk.LabelFrame(container, text="Fingerprint Upload", font=('Arial', 12, 'bold'),
                                   bg='white', padx=15, pady=15)
        left_panel.pack(side='left', fill='both', expand=True, padx=(0, 10))
        
        # Image preview
        self.image_label = tk.Label(left_panel, text="No image loaded", 
                                    bg='#ecf0f1', width=40, height=15)
        self.image_label.pack(pady=10)
        
        # Upload button
        tk.Button(left_panel, text="📁 Upload Fingerprint", command=self.upload_image,
                 font=('Arial', 12, 'bold'), bg='#3498db', fg='white',
                 padx=20, pady=10, cursor='hand2').pack(pady=10)
        
        # Detect button
        self.detect_btn = tk.Button(left_panel, text="🔍 Detect Blood Group", 
                                    command=self.detect_blood_group,
                                    font=('Arial', 12, 'bold'), bg='#27ae60', fg='white',
                                    padx=20, pady=10, cursor='hand2', state='disabled')
        self.detect_btn.pack(pady=10)
        
        # Right panel - Results
        right_panel = tk.LabelFrame(container, text="Detection Results", font=('Arial', 12, 'bold'),
                                    bg='white', padx=15, pady=15)
        right_panel.pack(side='right', fill='both', expand=True, padx=(10, 0))
        
        # Status
        self.status_label = tk.Label(right_panel, text="Status: Waiting for fingerprint...",
                                     font=('Arial', 11), bg='white', fg='#7f8c8d')
        self.status_label.pack(pady=10)
        
        # Result display
        self.result_frame = tk.Frame(right_panel, bg='white')
        self.result_frame.pack(fill='both', expand=True, pady=10)
        
        # Buttons
        btn_frame = tk.Frame(right_panel, bg='white')
        btn_frame.pack(pady=10)
        
        tk.Button(btn_frame, text="📊 View History", command=self.show_history,
                 font=('Arial', 10), bg='#9b59b6', fg='white',
                 padx=15, pady=8, cursor='hand2').pack(side='left', padx=5)
        
        tk.Button(btn_frame, text="📄 Generate Report", command=self.generate_report,
                 font=('Arial', 10), bg='#e67e22', fg='white',
                 padx=15, pady=8, cursor='hand2').pack(side='left', padx=5)
        
        self.current_image_path = None
        self.last_detection = None
    
    def upload_image(self):
        """Handle image upload"""
        file_path = filedialog.askopenfilename(
            title="Select Fingerprint Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        
        if file_path:
            self.current_image_path = file_path
            
            # Display image
            try:
                img = Image.open(file_path)
                img.thumbnail((300, 300))
                photo = ImageTk.PhotoImage(img)
                self.image_label.configure(image=photo, text="")
                self.image_label.image = photo
                
                self.detect_btn.config(state='normal')
                self.status_label.config(text="Status: Image loaded successfully", fg='#27ae60')
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def detect_blood_group(self):
        """Perform blood group detection"""
        if not self.current_image_path:
            messagebox.showerror("Error", "Please upload a fingerprint image first")
            return
        
        self.status_label.config(text="Status: Processing...", fg='#f39c12')
        self.root.update()
        
        try:
            # Preprocess image
            processed_img = preprocess_fingerprint(self.current_image_path)
            
            if processed_img is None:
                raise Exception("Failed to process image")
            
            # Make prediction
            predictions = self.model.predict(processed_img, verbose=0)
            predicted_class = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class] * 100
            
            blood_group = self.blood_groups[predicted_class]
            
            # Store detection
            detection_data = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'blood_group': blood_group,
                'confidence': float(confidence),
                'image_path': self.current_image_path
            }
            
            self.user_db.add_detection(self.current_user, detection_data)
            self.last_detection = detection_data
            
            # Display results
            self.display_results(blood_group, confidence, predictions[0])
            
            self.status_label.config(text="Status: Detection complete ✓", fg='#27ae60')
            
        except Exception as e:
            messagebox.showerror("Error", f"Detection failed: {str(e)}")
            self.status_label.config(text="Status: Detection failed ✗", fg='#e74c3c')
    
    def display_results(self, blood_group, confidence, all_predictions):
        """Display detection results"""
        # Clear previous results
        for widget in self.result_frame.winfo_children():
            widget.destroy()
        
        # Blood group result
        result_text = f"Detected Blood Group: {blood_group}"
        tk.Label(self.result_frame, text=result_text, font=('Arial', 16, 'bold'),
                bg='white', fg='#2c3e50').pack(pady=10)
        
        # Confidence
        confidence_text = f"Confidence: {confidence:.2f}%"
        tk.Label(self.result_frame, text=confidence_text, font=('Arial', 12),
                bg='white', fg='#27ae60').pack(pady=5)
        
        # Progress bar for confidence
        progress = ttk.Progressbar(self.result_frame, length=300, mode='determinate')
        progress['value'] = confidence
        progress.pack(pady=10)
        
        # All predictions chart
        fig, ax = plt.subplots(figsize=(5, 3))
        bars = ax.barh(self.blood_groups, all_predictions * 100)
        
        # Color the predicted group
        predicted_idx = np.argmax(all_predictions)
        bars[predicted_idx].set_color('#27ae60')
        
        ax.set_xlabel('Probability (%)')
        ax.set_title('All Blood Group Probabilities')
        plt.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, self.result_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(pady=10)
    
    def show_history(self):
        """Show detection history"""
        history_window = tk.Toplevel(self.root)
        history_window.title("Detection History")
        history_window.geometry("700x500")
        
        # Title
        tk.Label(history_window, text=f"Detection History - {self.current_user}",
                font=('Arial', 16, 'bold'), pady=10).pack()
        
        # History list
        frame = tk.Frame(history_window)
        frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Scrollbar
        scrollbar = tk.Scrollbar(frame)
        scrollbar.pack(side='right', fill='y')
        
        # Treeview
        columns = ('Date', 'Blood Group', 'Confidence')
        tree = ttk.Treeview(frame, columns=columns, show='headings', yscrollcommand=scrollbar.set)
        
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=200)
        
        # Add data
        detections = self.user_db.users[self.current_user]['detections']
        for detection in reversed(detections):
            tree.insert('', 'end', values=(
                detection['timestamp'],
                detection['blood_group'],
                f"{detection['confidence']:.2f}%"
            ))
        
        tree.pack(side='left', fill='both', expand=True)
        scrollbar.config(command=tree.yview)
    
    def generate_report(self):
        """Generate PDF report"""
        if not self.last_detection:
            messagebox.showwarning("Warning", "No detection to report. Please perform a detection first.")
            return
        
        try:
            # Create report content
            report_content = f"""
BLOOD GROUP DETECTION REPORT
{'=' * 50}

User: {self.current_user}
Date: {self.last_detection['timestamp']}

DETECTION RESULTS:
--------------------------------------------------
Blood Group: {self.last_detection['blood_group']}
Confidence: {self.last_detection['confidence']:.2f}%

Image: {os.path.basename(self.last_detection['image_path'])}

SYSTEM INFORMATION:
--------------------------------------------------
Model: Deep Learning CNN
Input: Fingerprint Image (128x128 pixels)
Processing: Automated feature extraction
Classification: 8-class blood group prediction

DISCLAIMER:
--------------------------------------------------
This is a research/educational system. Results should
be verified by professional medical testing. Do not
use for clinical diagnosis without proper validation.

{'=' * 50}
End of Report
"""
            
            # Save report
            filename = f"blood_group_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(filename, 'w') as f:
                f.write(report_content)
            
            messagebox.showinfo("Success", f"Report generated successfully!\nSaved as: {filename}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate report: {str(e)}")
    
    def logout(self):
        """Handle logout"""
        self.current_user = None
        self.show_login_screen()

# ==================== MAIN ====================

if __name__ == "__main__":
    root = tk.Tk()
    app = BloodGroupDetectorApp(root)
    root.mainloop()