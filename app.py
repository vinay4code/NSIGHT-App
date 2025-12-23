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
import bcrypt  # Secure password hashing

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
    page_title="N-SIGHT | Full-Stack ML Pipeline",
    layout="wide",
    page_icon="🔭",
    initial_sidebar_state="expanded"
)

# --- CSS STYLING ---
st.markdown("""
    <style>
    .stApp { background-color: #0b0f19; }
    h1, h2, h3 { font-family: 'Segoe UI', sans-serif; color: #e0e0e0; }
    
    /* Custom Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #FF4B4B 0%, #FF914D 100%);
        color: white; border: none; border-radius: 8px; font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(255, 75, 75, 0.4); }
    
    /* Metrics */
    div[data-testid="stMetric"] { background-color: #151a25; border: 1px solid #2a3142; border-radius: 8px; padding: 10px; }
    
    /* Expanders */
    div[data-testid="stExpander"] { border: 1px solid #2a3142; border-radius: 8px; background-color: #151a25; }
    </style>
""", unsafe_allow_html=True)

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
        st.error("⚠️ Database Offline. Cannot Register.")
        return
    users_ref = db.collection('users')
    if any(users_ref.where('email', '==', email).stream()):
        st.error("User already exists!")
        return
    users_ref.add({'email': email, 'password': hash_pass(password), 'role': role, 'created_at': datetime.datetime.now()})
    st.success("Account created! Please Login.")

def login_user(email, password):
    if not st.session_state.db_connected:
        # Offline Backdoor
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

# --- LOGIN SCREEN ---
if st.session_state.user is None:
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.markdown("<h1 style='text-align:center;'>🔭 N-SIGHT</h1>", unsafe_allow_html=True)
        tab1, tab2 = st.tabs(["Login", "Register"])
        with tab1:
            email = st.text_input("Email", key="l_email")
            password = st.text_input("Password", type="password", key="l_pass")
            if st.button("Login", use_container_width=True):
                user = login_user(email, password)
                if user: st.session_state.user = user; st.rerun()
                else: st.error("Invalid credentials (Try 'demo'/'demo' if offline)")
        with tab2:
            r_email = st.text_input("Email", key="r_email")
            r_pass = st.text_input("Password", type="password", key="r_pass")
            r_role = st.selectbox("Role", ["Student", "Admin"])
            if st.button("Register", use_container_width=True):
                register_user(r_email, r_pass, r_role)
        
        if not st.session_state.db_connected: st.info("💡 Offline Mode Active")

# --- DASHBOARD ---
else:
    # SIDEBAR
    with st.sidebar:
        st.markdown(f"**👤 {st.session_state.user['role']}:** {st.session_state.user['email']}")
        if st.button("Logout"): st.session_state.user = None; st.rerun()
        st.divider()
        input_source = st.radio("Source", ["Upload File", "Live Camera", "Simulation"])
        st.divider()
        with st.expander("⚙️ Calibration"):
            start_wl = st.number_input("Start Å", 4000.0, step=100.0)
            disp = st.number_input("Å/px", 1.5, step=0.1)
        with st.expander("🎛️ Processing"):
            smooth = st.slider("Smoothing", 1, 21, 5, 2)
            deriv = st.selectbox("Derivative", [0, 1, 2], format_func=lambda x: ["Raw","Slope","Curve"][x])

    # MAIN HEADER
    c1, c2 = st.columns([3,1])
    c1.title("Spectral Analysis Dashboard")
    c2.metric("DB Status", "Online" if st.session_state.db_connected else "Offline", delta_color="normal")

    # INPUT HANDLING
    data = None
    if input_source == "Upload File":
        f = st.file_uploader("Upload .fits/.ser", type=["fit", "fits", "ser"])
        if f:
            if f.name.endswith('.ser'):
                r = SERReader(f)
                if st.button("Stack 50 Frames"): 
                    data = np.mean([r.get_frame(i) for i in range(min(50, r.header['FrameCount']))], axis=0)
                else: data = r.get_frame(0)
            else:
                with fits.open(f) as h: data = h[0].data
    elif input_source == "Live Camera":
        if st.button("Toggle Camera"): st.session_state.run_cam = not st.session_state.run_cam
        if st.session_state.run_cam:
            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()
            if ret:
                st.image(frame, channels="BGR")
                if st.button("Capture"):
                    st.session_state.captured_data = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    st.session_state.run_cam = False; st.rerun()
            cap.release()
        if st.session_state.captured_data is not None: data = st.session_state.captured_data
    else:
        x = np.linspace(4000, 7000, 1000)
        data = 100 + (x-4000)*0.03 + 500*np.exp(-0.5*((x-6563)/10)**2) + np.random.normal(0, 3, 1000)

    # PROCESS DATA
    if data is not None:
        if data.ndim == 2: flux = np.mean(data[data.shape[0]//2-10:data.shape[0]//2+10, :], axis=0)
        else: flux = data
        
        # Apply Smoothing & Deriv
        if smooth > 1:
            if smooth % 2 == 0: smooth += 1
            flux = savgol_filter(flux, smooth, 3, deriv=deriv)
        
        x_axis = start_wl + (np.arange(len(flux)) * disp)

        # --- TABS: The Core of the Full Stack App ---
        tabs = ["📊 Analyze", "🧠 ML Pipeline", "💾 My Library"]
        if st.session_state.user['role'] == "Admin": tabs.append("🛡️ Admin")
        curr_tab = st.radio("Tab", tabs, horizontal=True, label_visibility="collapsed")
        st.divider()

        # 1. ANALYZE TAB (Visualize + Create)
        if curr_tab == "📊 Analyze":
            fig = go.Figure(go.Scatter(x=x_axis, y=flux, line=dict(color='#00F0FF')))
            fig.update_layout(template="plotly_dark", height=400, title="Spectrum View")
            st.plotly_chart(fig, use_container_width=True)
            
            c1, c2 = st.columns([3, 1])
            s_name = c1.text_input("Label for Cloud Save", "Observation 1")
            if c2.button("💾 Save to DB"):
                save_spectrum(s_name, x_axis, flux, st.session_state.user['email'])

        # 2. ML PIPELINE TAB (Research Paper Feature)
        elif curr_tab == "🧠 ML Pipeline":
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("1. Build Dataset")
                label = st.text_input("Class Label (e.g. Star Type A)")
                if st.button("➕ Add Current to Dataset"):
                    _, rs_flux = resample_spectrum(x_axis, normalize_data(flux))
                    row = pd.DataFrame([np.append([label], rs_flux)], columns=['label'] + [f'px_{i}' for i in range(1000)])
                    st.session_state.dataset = pd.concat([st.session_state.dataset, row], ignore_index=True)
                    st.success(f"Added! Total samples: {len(st.session_state.dataset)}")
                st.dataframe(st.session_state.dataset.head(3), height=100)

            with c2:
                st.subheader("2. Train & Predict")
                if st.button("🚀 Train Random Forest"):
                    if len(st.session_state.dataset) > 1:
                        df = st.session_state.dataset
                        X, y = df.drop('label', axis=1), df['label']
                        clf = RandomForestClassifier(n_estimators=100)
                        clf.fit(X, y)
                        st.session_state.trained_model = clf
                        st.success("Model Trained!")
                    else: st.warning("Need more data.")
                
                if st.session_state.trained_model:
                    if st.button("🔮 Predict Current Spectrum"):
                        _, input_vec = resample_spectrum(x_axis, normalize_data(flux))
                        pred = st.session_state.trained_model.predict([input_vec])[0]
                        prob = np.max(st.session_state.trained_model.predict_proba([input_vec]))
                        st.metric("Prediction", pred, f"{prob:.1%} Conf")

        # 3. LIBRARY TAB (Read, Update, Delete)
        elif curr_tab == "💾 My Library":
            search = st.text_input("🔍 Search Library...", "")
            if st.session_state.db_connected:
                docs = db.collection('spectra').where('user_id', '==', st.session_state.user['email']).stream()
                for doc in docs:
                    d = doc.to_dict()
                    if search.lower() in d['label'].lower():
                        with st.expander(f"📄 {d['label']} ({d['timestamp'].strftime('%Y-%m-%d')})"):
                            # Update Logic
                            new_label = st.text_input("Edit Label", d['label'], key=f"edit_{doc.id}")
                            if st.button("Update", key=f"up_{doc.id}"):
                                update_spectrum(doc.id, new_label); st.rerun()
                            # Delete Logic
                            if st.button("Delete", key=f"del_{doc.id}"):
                                delete_spectrum(doc.id); st.rerun()
                            # Visualization
                            st.line_chart(d['flux'], height=100)
            else: st.info("Connect DB to view library.")

        # 4. ADMIN TAB (RBAC)
        elif curr_tab == "🛡️ Admin":
            if st.session_state.db_connected:
                u_count = len(list(db.collection('users').stream()))
                s_count = len(list(db.collection('spectra').stream()))
                c1, c2 = st.columns(2)
                c1.metric("Total Users", u_count)
                c2.metric("Total Spectra", s_count)
                st.write("All Data Access:")
                all_data = [{"User": d.to_dict()['user_id'], "Label": d.to_dict()['label']} for d in db.collection('spectra').stream()]
                st.dataframe(all_data)