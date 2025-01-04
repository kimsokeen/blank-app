import streamlit as st
import sqlite3
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
#import cv2

def iou_metric(y_true, y_pred):
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true + y_pred) - intersection
    return intersection / (union + tf.keras.backend.epsilon())

segmentation_model = load_model('best_model3.keras', custom_objects={'iou_metric': iou_metric})
model = load_model('mobilenet_model1.h5')

# Compile the model if needed
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy', 'Precision', 'Recall'])

# Set up Streamlit configuration
st.set_page_config(layout="centered", page_title="Mobile Web App", page_icon="📱")

# Initialize page state if not set
if "page" not in st.session_state:
    st.session_state.page = 1  # Start from page 1

# Function to navigate to the specified page with single click
def navigate(page):
    st.session_state.page = page

# Function to display a "Back" button
def back_button(destination_page):
    st.markdown(
        """
        <style>
            .stButtonSmall button {{
                width: 60px;
                height: 35px;
                font-size: 14px;
                background-color: #fff;
                color: #333;
                border: 1px solid #333;
                border-radius: 5px;
                font-weight: bold;
            }}
            .stButtonSmall button:hover {{
                background-color: #e6e6e6;
            }}
        </style>
        """,
        unsafe_allow_html=True
    )
    if st.button("← Back", key=f"back_{destination_page}"):
        navigate(destination_page)

def setup_database():
    conn = sqlite3.connect('user_data.db')
    c = conn.cursor()
    # Create or update the users table with additional columns
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY, 
            password TEXT,
            full_name TEXT,
            age INTEGER,
            email TEXT
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS results (
            username TEXT,
            result TEXT,
            wound_size REAL,
            timestamp TEXT
        )
    ''')
    conn.commit()
    conn.close()

def migrate_database():
    conn = sqlite3.connect('user_data.db')
    c = conn.cursor()
    
    # Check columns in the 'users' table
    c.execute("PRAGMA table_info(users)")
    columns = [col[1] for col in c.fetchall()]
    
    if 'full_name' not in columns:
        c.execute("ALTER TABLE users ADD COLUMN full_name TEXT")
        print("Database updated: 'full_name' column added.")
    
    if 'profile_pic' not in columns:
        c.execute("ALTER TABLE users ADD COLUMN profile_pic TEXT")
        print("Database updated: 'profile_pic' column added.")
    
    # Check columns in the 'results' table
    c.execute("PRAGMA table_info(results)")
    columns = [col[1] for col in c.fetchall()]
    
    if 'wound_size' not in columns:
        c.execute("ALTER TABLE results ADD COLUMN wound_size REAL")
        print("Database updated: 'wound_size' column added.")
    
    conn.commit()
    conn.close()

# Locate the database connection section (near the start of your script)
conn = sqlite3.connect('user_data.db')
c = conn.cursor()

# Add this to ensure the database schema is updated
migrate_database()

def update_schema():
    conn = sqlite3.connect('user_data.db')
    c = conn.cursor()

    # Check if the 'age' column exists
    c.execute("PRAGMA table_info(users)")
    columns = [info[1] for info in c.fetchall()]
    if 'age' not in columns:
        print("Updating database schema...")
        # Drop and recreate the table (Option 1)
        c.execute("DROP TABLE IF EXISTS users")
        c.execute("""
        CREATE TABLE users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            password TEXT NOT NULL,
            full_name TEXT NOT NULL,
            age INTEGER,
            email TEXT NOT NULL
        )
        """)
        conn.commit()
        print("Schema updated successfully.")
    else:
        print("Schema already up to date.")

    conn.close()


def create_account(username, password, name, age, email):
    conn = sqlite3.connect('user_data.db')
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password, full_name, age, email) VALUES (?, ?, ?, ?, ?)", 
                  (username, password, name, age, email))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


def authenticate_user(username, password):
    conn = sqlite3.connect('user_data.db')
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, password))
    result = c.fetchone()
    conn.close()
    return result is not None

def save_result(username, result, wound_size):
    conn = sqlite3.connect('user_data.db')
    c = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO results (username, timestamp, result, wound_size) VALUES (?, ?, ?, ?)",
              (username, timestamp, result, wound_size))
    conn.commit()
    conn.close()


def get_latest_results(username):
    conn = sqlite3.connect('user_data.db')
    c = conn.cursor()
    c.execute("SELECT timestamp, result, wound_size FROM results WHERE username = ? ORDER BY timestamp DESC LIMIT 10", (username,))
    results = c.fetchall()
    conn.close()
    return results

def get_user_info(username):
    conn = sqlite3.connect('user_data.db')
    c = conn.cursor()
    c.execute("SELECT full_name, age, email, profile_pic FROM users WHERE username = ?", (username,))
    info = c.fetchone()
    conn.close()
    return info

def analyze_image(image_data):
    # Classification
    img = image_data.convert("RGB")  # Convert to RGB for 3 channels
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Ensure the image is RGB (remove alpha channel if present)
    if img.mode == 'RGBA':
        img = img.convert('RGB')

    classification_prediction = model.predict(img_array)
    probability = classification_prediction[0][0]  # Probability for class "normal"
    predicted_class = "non-diabetic foot" if probability > 0.5 else "diabetic foot ulcers"

    # Segmentation
    seg_img = image_data.resize((256, 256))  # Resize for segmentation input
    seg_array = np.array(seg_img) / 255.0
    seg_array = np.expand_dims(seg_array, axis=0)

    segmentation_mask = segmentation_model.predict(seg_array)[0]

    # Debugging: Visualize raw segmentation mask
    st.image(segmentation_mask, caption="Raw Segmentation Mask", clamp=True)

    # Thresholding to create binary mask
    segmentation_mask = (segmentation_mask > 0.3).astype(np.uint8)

    # Debugging: Visualize thresholded mask
    st.image(segmentation_mask * 255, caption="Thresholded Mask", clamp=True)

    wound_size = np.sum(segmentation_mask)  # Count pixels as wound size

    # Debugging: Output wound size
    st.write(f"Wound Size (pixels): {wound_size}")

    # Overlay mask on the original image
    original_image = np.array(image_data.convert("RGB"))  # Ensure 3-channel RGB format
    segmentation_mask_resized = cv2.resize(segmentation_mask, (original_image.shape[1], original_image.shape[0]))

    # Ensure the segmentation mask has the same number of channels as the original image
    if len(segmentation_mask_resized.shape) == 2:
        segmentation_mask_resized = np.stack([segmentation_mask_resized] * 3, axis=-1)

    overlay = cv2.addWeighted(original_image, 0.7, segmentation_mask_resized * 255, 0.3, 0)

    return predicted_class, probability, wound_size, overlay


# Enhanced UI design for main page buttons
def main_page_buttons():
    # CSS for larger buttons
    st.markdown("""
        <style>
        .big-button {
            display: inline-block;
            width: 100%;
            height: 100px;  /* Adjusted for prominence */
            font-size: 20px;
            text-align: center;
            line-height: 100px;  /* Center text vertically */
            background-color: #4CAF50;
            color: white;
            border-radius: 10px;
            margin: 10px 0;
            font-weight: bold;
            cursor: pointer;
            text-decoration: none;
        }
        .big-button:hover {
            background-color: #45a049;
        }
        </style>
        """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<a href="#" class="big-button" onclick="window.parent.location.href=\'#page6\';">Upload Photo</a>', unsafe_allow_html=True)
    with col2:
        st.markdown('<a href="#" class="big-button" onclick="window.parent.location.href=\'#page7\';">Talk to Doctor</a>', unsafe_allow_html=True)


# Page 1: Get Started
def page1():
    st.title("Ulcers Detector")
    if st.button("Get Started"):
        navigate(2)

# Page 2: Log In
def page2():
    back_button(1)
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Log in"):
        if not username or not password:
            st.error("Please enter both username and password.")
        elif authenticate_user(username, password):
            st.session_state.username = username
            navigate(5)
        else:
            st.error("Invalid credentials.")
    
    if st.button("Create an Account"):
        st.session_state.new_username = username
        st.session_state.new_password = password
        navigate(3)


# Page 3: Account Creation
def page3():
    back_button(2)
    st.title("Create Account")
    username = st.text_input("New Username", value=st.session_state.get("new_username", ""))
    password = st.text_input("New Password", type="password", value=st.session_state.get("new_password", ""))
    name = st.text_input("Full Name")
    age = st.number_input("Age", min_value=1, step=1)
    email = st.text_input("Email")
    if st.button("Create Account"):
        if not username or not password:
            st.error("Please fill in all required fields.")
        elif create_account(username, password, name, age, email):
            st.success("Account created successfully!")
            st.session_state.username = username
            navigate(5)
        else:
            st.error("Username already exists. Please choose a different one.")
        

# Page 5: Main page
def page5():
    st.title(f"Hi, {st.session_state.username}.")
    st.write("Welcome to your personal assistance.")
    
    results = get_latest_results(st.session_state.username)
    if results:
        st.write("Recent Analysis Results:")
        for timestamp, result, wound_size in results:
            st.write(f"Timestamp: {timestamp}")
            st.write(f"Result: {result}")
            st.write(f"Wound Size: {wound_size}")
            st.write("---")
    else:
        st.write("No results found.")
    
    if st.button("Upload Photo"):
        navigate(6)
    st.write("")  # Adds a bit of vertical space
    if st.button("Talk to Doctor"):
        navigate(7)  # Implement the action you want for this button
    st.write("")
    if st.button("Trend Analysis"):
        pass
    st.write("")
    if st.button("User Account"):
        navigate(8)

# Page6 : Analyze
def page6():
    back_button(5)
    st.title("Upload and Analyze")
    uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_image is not None:
        image_data = Image.open(uploaded_image)
        st.image(image_data, caption="Uploaded Image", use_column_width=True)
        
        if st.button("Analyze"):
            predicted_class, probability, wound_size, overlay = analyze_image(image_data)
            probability = round(probability, 4)
            wound_size_cm = wound_size * 0.01  # Example scaling to cm²
            
            # Display results
            st.success(f"Classification result: {predicted_class}")
            st.info(f"Estimated wound size: {wound_size_cm:.2f} cm²")
            st.image(overlay, caption="Segmentation Result", use_column_width=True)
            
            # Save results
            save_result(st.session_state.username, predicted_class, wound_size_cm)
            
            if st.button("Finish"):
                navigate(5)

# Page 7: Talk to Doctor
def page7():
    back_button(5)
    st.title("Talk to Doctor")
    st.write("Chat with our experts")
    
    chat_history = st.session_state.get("chat_history", [])
    
    for message in chat_history:
        st.write(f"{message['sender']}: {message['content']}")
    
    user_input = st.text_input("Your message")
    if st.button("Send"):
        chat_history.append({"sender": "User", "content": user_input})
        st.session_state.chat_history = chat_history
        st.write(f"You: {user_input}")

# Page 8: User Account
def page8():
    back_button(5)
    st.title("User Account")
    
    user_info = get_user_info(st.session_state.username)
    if user_info:
        full_name, age, email, profile_pic = user_info
        
        st.image(profile_pic, caption="Profile Picture", width=150) if profile_pic else st.write("No profile picture")
        st.write(f"**Full Name**: {full_name}")
        st.write(f"**Age**: {age}")
        st.write(f"**Email**: {email}")
        
        if st.button("Edit"):
            st.write("Edit functionality not implemented yet.")
    else:
        st.write("No user info found")


# Main function
def main():
    setup_database()
    update_schema()
    
    if st.session_state.page == 1:
        page1()
    elif st.session_state.page == 2:
        page2()
    elif st.session_state.page == 3:
        page3()
    elif st.session_state.page == 5:
        page5()
    elif st.session_state.page == 6:
        page6()
    elif st.session_state.page == 7:
        page7()
    elif st.session_state.page == 8:
        page8()

if __name__ == "__main__":
    main()
