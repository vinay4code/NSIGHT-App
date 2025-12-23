import streamlit as st
import numpy as np
import pandas as pd
import cv2
from astropy.io import fits
import plotly.graph_objects as go
from scipy.signal import find_peaks, savgol_filter
from scipy.signal.windows import gaussian
from scipy.ndimage import convolve1d
from scipy.interpolate import interp1d
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import struct
from PIL import Image
import datetime
import time
import bcrypt

# --- FIREBASE SETUP ---
import firebase_admin
from firebase_admin import credentials, firestore

# Initialize Firebase
if not firebase_admin._apps:
    try:
        # Check if secrets file is found
        if 'firebase' in st.secrets:
            cred_info = {k:v for k,v in st.secrets["firebase"].items()}
            cred = credentials.Certificate(cred_info)
            firebase_admin.initialize_app(cred)
            db = firestore.client()
            st.session_state.db_connected = True
        else:
            st.error("❌ Error: Could not find [firebase] section in secrets.toml")
            st.session_state.db_connected = False
    except Exception as e:
        st.error(f"❌ Connection Failed: {e}") # This will print the exact error on screen
        st.session_state.db_connected = False
else:
    db = firestore.client()
    st.session_state.db_connected = True

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="N-SIGHT | Full-Stack ML",
    layout="wide",
    page_icon="🔭",
    initial_sidebar_state="expanded"
)

# --- ADVANCED UI STYLING (Mobile & Desktop) ---
def apply_custom_style():
    st.markdown("""
        <style>
        /* --- GLOBAL VARIABLES --- */
        :root {
            --primary: #00F0FF;
            --bg-glass: rgba(17, 22, 32, 0.7);
            --border-glass: rgba(255, 255, 255, 0.1);
        }

        /* --- MOBILE OPTIMIZATION --- */
        /* Reduce the massive top padding Streamlit adds by default */
        .block-container {
            padding-top: 2rem !important;
            padding-bottom: 5rem !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
        
        /* Hide the default hamburger menu on mobile for a cleaner look */
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}

        /* --- GLASSMORPHISM CARDS --- */
        div[data-testid="stMetric"], div[data-testid="stExpander"] {
            background: var(--bg-glass);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border: 1px solid var(--border-glass);
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            transition: transform 0.2s ease;
        }
        
        div[data-testid="stMetric"]:hover {
            border-color: var(--primary);
            transform: translateY(-2px);
        }

        /* --- TYPOGRAPHY --- */
        h1 { font-weight: 800 !important; letter-spacing: -1px; background: -webkit-linear-gradient(0deg, #fff, #999); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        h2, h3 { font-weight: 600 !important; color: #eee !important; }
        p, label { color: #bbb !important; }

        /* --- BUTTONS --- */
        .stButton>button {
            background: linear-gradient(135deg, #00F0FF 0%, #0061ff 100%);
            color: #000;
            border: none;
            border-radius: 8px;
            font-weight: 700;
            height: 3rem;
            width: 100%; /* Full width for mobile touch targets */
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            box-shadow: 0 4px 15px rgba(0, 240, 255, 0.4);
            color: #fff;
        }

        /* --- INPUT FIELDS --- */
        input {
            background-color: #0d1117 !important;
            color: #fff !important;
            border: 1px solid #30363d !important;
            border-radius: 8px !important;
        }
        
        /* --- LOGIN CARD CONTAINER --- */
        .login-container {
            background: #111620;
            padding: 2rem;
            border-radius: 16px;
            border: 1px solid #30363d;
            box-shadow: 0 10px 40px rgba(0,0,0,0.5);
            margin-top: 10vh;
        }
        </style>
    """, unsafe_allow_html=True)

apply_custom_style()

# --- HELPER CLASSES & FUNCTIONS ---
class SERReader:
    def __init__(self, file_buffer):
        self.file = file_buffer
        self.header = {}
        self._parse_header()
    def _parse_header(self):
        self.file.seek(0)
        header_data = self.file.read(178)
        self.header['FrameCount'] = struct.unpack('<I', header_data[38:42])[0]
        self.header['Width'] = struct.unpack('<I', header_data[26:30])[0]
        self.header['Height'] = struct.unpack('<I', header_data[30:34])[0]
        self.header['PixelDepth'] = struct.unpack('<I', header_data[34:38])[0]
        self.bytes_per_pixel = 1 if self.header['PixelDepth'] <= 8 else 2
        self.frame_size = self.header['Width'] * self.header['Height'] * self.bytes_per_pixel
    def get_frame(self, frame_index):
        if frame_index >= self.header['FrameCount']: return None
        offset = 178 + (frame_index * self.frame_size)
        self.file.seek(offset)
        data = self.file.read(self.frame_size)
        dtype = np.uint8 if self.bytes_per_pixel == 1 else np.uint16
        return np.frombuffer(data, dtype=dtype).reshape((self.header['Height'], self.header['Width']))

COMMON_LINES = {"Hydrogen": {"Hα": 6562.8, "Hβ": 4861.3}, "Helium": {"He I": 5875.6}, "Sodium": {"Na D": 5890.0}}

def normalize_data(data): return (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-6)

def resample_spectrum(wavelengths, flux, target_start=4000, target_end=7000, n_points=1000):
    target_wave = np.linspace(target_start, target_end, n_points)
    f = interp1d(wavelengths, flux, kind='linear', fill_value="extrapolate")
    return target_wave, f(target_wave)

# --- AUTH & DB FUNCTIONS ---
def hash_pass(password): return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
def check_pass(password, hashed): return bcrypt.checkpw(password.encode(), hashed.encode())

def register_user(email, password, role="Student"):
    if not st.session_state.db_connected:
        st.error("⚠️ Offline Mode. Cannot Register.")
        return
    users_ref = db.collection('users')
    if any(users_ref.where('email', '==', email).stream()):
        st.error("User exists!")
        return
    users_ref.add({'email': email, 'password': hash_pass(password), 'role': role, 'created_at': datetime.datetime.now()})
    st.success("Account created! Login now.")

def login_user(email, password):
    if not st.session_state.db_connected:
        if email == "demo" and password == "demo": return {"email": "demo", "role": "Student", "id": "local"}
        if email == "admin" and password == "admin": return {"email": "admin", "role": "Admin", "id": "admin"}
        return None
    users_ref = db.collection('users')
    for doc in users_ref.where('email', '==', email).stream():
        u = doc.to_dict()
        if check_pass(password, u['password']):
            u['id'] = doc.id
            return u
    return None

def save_spectrum(label, wav, flux, user_email):
    if not st.session_state.db_connected:
        st.warning("Offline: Saved locally.")
        return
    w_s, f_s = resample_spectrum(wav, flux, n_points=500)
    db.collection('spectra').add({
        'user_id': user_email, 'label': label,
        'wavelengths': w_s.tolist(), 'flux': f_s.tolist(),
        'timestamp': datetime.datetime.now()
    })
    st.toast("✅ Saved to Cloud")

def update_spectrum(doc_id, new_label):
    if st.session_state.db_connected:
        db.collection('spectra').document(doc_id).update({'label': new_label})
        st.toast("✏️ Updated")

def delete_spectrum(doc_id):
    if st.session_state.db_connected:
        db.collection('spectra').document(doc_id).delete()
        st.toast("🗑️ Deleted")

# --- SESSION STATE ---
if 'user' not in st.session_state: st.session_state.user = None
if 'dataset' not in st.session_state: st.session_state.dataset = pd.DataFrame(columns=['label'] + [f'px_{i}' for i in range(1000)])
if 'trained_model' not in st.session_state: st.session_state.trained_model = None
if 'captured_data' not in st.session_state: st.session_state.captured_data = None
if 'run_cam' not in st.session_state: st.session_state.run_cam = False

# =========================================================
#                       MAIN APP
# =========================================================

# --- 1. LOGIN SCREEN (Redesigned for Mobile/Desktop) ---
if st.session_state.user is None:
    # Use empty columns to center the login box on desktop
    # On mobile, columns stack automatically
    col_l, col_center, col_r = st.columns([1, 4, 1])
    
    with col_center:
        st.markdown("<div style='height: 5vh;'></div>", unsafe_allow_html=True) # Spacer
        st.markdown("""
        <div style='text-align: center; margin-bottom: 20px;'>
            <h1 style='font-size: 3rem;'>🔭 N-SIGHT</h1>
            <p>Full-Stack Spectral Intelligence</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Login "Card" UI
        with st.container():
            tab_login, tab_reg = st.tabs(["Login", "Sign Up"])
            
            with tab_login:
                email = st.text_input("Email Address", key="l_email")
                password = st.text_input("Password", type="password", key="l_pass")
                st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
                if st.button("🚀 Login", use_container_width=True):
                    user = login_user(email, password)
                    if user: st.session_state.user = user; st.rerun()
                    else: st.error("Invalid Login")
            
            with tab_reg:
                r_email = st.text_input("Email", key="r_email")
                r_pass = st.text_input("Password", type="password", key="r_pass")
                r_role = st.selectbox("Role", ["Student", "Admin"])
                st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
                if st.button("✨ Create Account", use_container_width=True):
                    register_user(r_email, r_pass, r_role)

        if not st.session_state.db_connected: 
            st.warning("⚠️ Offline Mode: Features limited.")

# --- 2. MAIN DASHBOARD ---
else:
    # SIDEBAR
    with st.sidebar:
        st.image("Nakshatra_transparent_1.png", width=None)
        st.markdown(f"**👤 {st.session_state.user['role']}**")
        if st.button("Logout", use_container_width=True): st.session_state.user = None; st.rerun()
        st.divider()
        
        input_source = st.radio("📡 Data Source", ["Upload File", "Live Camera", "Simulation"])
        
        with st.expander("⚙️ Calibration Settings"):
            start_wl = st.number_input("Start Å", 4000.0, step=100.0)
            disp = st.number_input("Å/px", 1.5, step=0.1)
        with st.expander("🎛️ Signal Processing"):
            smooth = st.slider("Smoothing", 1, 21, 5, 2)
            deriv = st.selectbox("Derivative", [0, 1, 2], format_func=lambda x: ["Raw","Slope","Curve"][x])

    # HEADER
    c1, c2 = st.columns([3,1])
    c1.title("Spectral Dashboard")
    c2.metric("System Status", "Online" if st.session_state.db_connected else "Local", delta_color="normal")

    # DATA INGESTION
    data = None
    if input_source == "Upload File":
        f = st.file_uploader("Upload Spectral Data", type=["fit", "fits", "ser"])
        if f:
            if f.name.endswith('.ser'):
                r = SERReader(f)
                if st.button("Stack 50 Frames", use_container_width=True): 
                    data = np.mean([r.get_frame(i) for i in range(min(50, r.header['FrameCount']))], axis=0)
                else: data = r.get_frame(0)
            else:
                with fits.open(f) as h: data = h[0].data
    elif input_source == "Live Camera":
        if st.button("Start/Stop Camera", use_container_width=True): st.session_state.run_cam = not st.session_state.run_cam
        if st.session_state.run_cam:
            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()
            if ret:
                st.image(frame, channels="BGR", use_container_width=True)
                if st.button("📸 Capture Frame", use_container_width=True):
                    st.session_state.captured_data = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    st.session_state.run_cam = False; st.rerun()
            cap.release()
        if st.session_state.captured_data is not None: data = st.session_state.captured_data
    else:
        x = np.linspace(4000, 7000, 1000)
        data = 100 + (x-4000)*0.03 + 500*np.exp(-0.5*((x-6563)/10)**2) + np.random.normal(0, 3, 1000)

    # VISUALIZATION & LOGIC
    if data is not None:
        if data.ndim == 2: flux = np.mean(data[data.shape[0]//2-10:data.shape[0]//2+10, :], axis=0)
        else: flux = data
        
        if smooth > 1:
            if smooth % 2 == 0: smooth += 1
            flux = savgol_filter(flux, smooth, 3, deriv=deriv)
        
        x_axis = start_wl + (np.arange(len(flux)) * disp)

        # TABS
        tabs = ["📊 Analyze", "🧠 ML Lab", "💾 Library"]
        if st.session_state.user['role'] == "Admin": tabs.append("🛡️ Admin")
        curr_tab = st.radio("Nav", tabs, horizontal=True, label_visibility="collapsed")
        st.divider()

        if curr_tab == "📊 Analyze":
            # Plotly chart with 0 margins for mobile
            fig = go.Figure(go.Scatter(x=x_axis, y=flux, line=dict(color='#00F0FF', width=2), fill='tozeroy'))
            fig.update_layout(template="plotly_dark", height=350, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
            
            with st.expander("Save Analysis"):
                s_name = st.text_input("Label", "Observation 1")
                if st.button("Save to Cloud", use_container_width=True):
                    save_spectrum(s_name, x_axis, flux, st.session_state.user['email'])

        elif curr_tab == "🧠 ML Lab":
            c1, c2 = st.columns(2)
            with c1:
                st.info("1. Build Dataset")
                label = st.text_input("Class Label")
                if st.button("Add Sample", use_container_width=True):
                    _, rs_flux = resample_spectrum(x_axis, normalize_data(flux))
                    row = pd.DataFrame([np.append([label], rs_flux)], columns=['label'] + [f'px_{i}' for i in range(1000)])
                    st.session_state.dataset = pd.concat([st.session_state.dataset, row], ignore_index=True)
                    st.success(f"Count: {len(st.session_state.dataset)}")
            with c2:
                st.info("2. Train & Test")
                if st.button("Train Model", use_container_width=True):
                    if len(st.session_state.dataset) > 1:
                        df = st.session_state.dataset
                        X, y = df.drop('label', axis=1), df['label']
                        clf = RandomForestClassifier(n_estimators=100)
                        clf.fit(X, y)
                        st.session_state.trained_model = clf
                        st.success("Trained!")
                    else: st.warning("Need Data")
                
                if st.session_state.trained_model:
                    if st.button("Predict Live", use_container_width=True):
                        _, input_vec = resample_spectrum(x_axis, normalize_data(flux))
                        pred = st.session_state.trained_model.predict([input_vec])[0]
                        st.metric("Result", pred)

        elif curr_tab == "💾 Library":
            search = st.text_input("🔍 Filter...", "")
            if st.session_state.db_connected:
                docs = db.collection('spectra').where('user_id', '==', st.session_state.user['email']).stream()
                for doc in docs:
                    d = doc.to_dict()
                    if search.lower() in d['label'].lower():
                        with st.expander(f"{d['label']}"):
                            new_label = st.text_input("Edit", d['label'], key=f"edit_{doc.id}")
                            c_up, c_del = st.columns(2)
                            if c_up.button("Save", key=f"up_{doc.id}", use_container_width=True):
                                update_spectrum(doc.id, new_label); st.rerun()
                            if c_del.button("Delete", key=f"del_{doc.id}", use_container_width=True):
                                delete_spectrum(doc.id); st.rerun()
                            st.line_chart(d['flux'], height=100)

        elif curr_tab == "🛡️ Admin":
            if st.session_state.db_connected:
                u_count = len(list(db.collection('users').stream()))
                s_count = len(list(db.collection('spectra').stream()))
                c1, c2 = st.columns(2)
                c1.metric("Users", u_count)
                c2.metric("Records", s_count)
                st.dataframe([{"User": d.to_dict()['user_id'], "Label": d.to_dict()['label']} for d in db.collection('spectra').stream()], use_container_width=True)