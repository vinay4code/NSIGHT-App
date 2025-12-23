import streamlit as st
import numpy as np
import pandas as pd
from astropy.io import fits
from PIL import Image
import plotly.graph_objects as go
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from sklearn.ensemble import RandomForestClassifier
import struct
import datetime
import bcrypt

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="N-SIGHT | Spectral Intelligence Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= STYLING =================
def apply_custom_style():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

    :root {
        --bg-main:#0B0F14;
        --bg-card:rgba(20,25,35,.75);
        --border:rgba(255,255,255,.08);
        --text:#E6EDF3;
        --muted:#9BA3AF;
        --accent:#4FC3F7;
    }

    html, body { background:var(--bg-main); color:var(--text); font-family:Inter,sans-serif; }

    /* MAIN CONTENT ONLY */
    .block-container { padding-top:2rem; padding-bottom:4rem; }

    h1,h2,h3 { font-weight:700; }
    p,label { color:var(--muted); }

    /* CARDS (NOT SIDEBAR) */
    div[data-testid="stMetric"], 
    div[data-testid="stExpander"] {
        background:var(--bg-card);
        border:1px solid var(--border);
        border-radius:14px;
    }

    /* INPUTS */
    input,select,textarea {
        background:#0F141C!important;
        color:var(--text)!important;
        border:1px solid var(--border)!important;
        border-radius:8px!important;
    }

    /* SIDEBAR FIX */
    section[data-testid="stSidebar"] {
        background:#0F141C;
        min-width:280px!important;
        max-width:280px!important;
    }

    section[data-testid="stSidebar"] > div {
        padding-top:1rem!important;
    }
    /* ================= LOGIN CENTERING ================= */
    
    .login-wrapper {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100vh;
    }
    
    .login-card {
        width: 420px;
        padding: 2.5rem;
        background: rgba(20, 25, 35, 0.85);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        box-shadow: 0 20px 60px rgba(0,0,0,0.6);
    }
    
    .login-title {
        text-align: center;
        margin-bottom: 1.5rem;
        font-size: 2.2rem;
        letter-spacing: 2px;
    }


    </style>
    """, unsafe_allow_html=True)

apply_custom_style()

# ================= SESSION =================
for k, v in {
    "user": None,
    "data_ready": False,
    "captured_data": None,
    "dataset": pd.DataFrame(columns=["label"]+[f"px_{i}" for i in range(1000)]),
    "trained_model": None
}.items():
    st.session_state.setdefault(k, v)
# Default calibration (safe fallback)
DEFAULT_START_WL = 4000.0
DEFAULT_DISP = 1.5


# ================= HELPERS =================
def normalize(x): return (x-np.min(x))/(np.max(x)-np.min(x)+1e-6)

def resample(w,f,n=1000):
    t=np.linspace(4000,7000,n)
    return t, interp1d(w,f,fill_value="extrapolate")(t)

# ================= LOGIN (SHORTENED) =================
if st.session_state.user is None:

    st.markdown("<br><br><br><br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1.2, 1])

    with col2:
        st.markdown(
            "<h1 style='text-align:center; letter-spacing:3px;'>N-SIGHT</h1>",
            unsafe_allow_html=True
        )

        email = st.text_input("Email")
        password = st.text_input("Password", type="password")

        if st.button("Login", use_container_width=True):
            st.session_state.user = {"email": email, "role": "Admin"}
            st.rerun()

    st.stop()


# ================= SIDEBAR (FIRST) =================
with st.sidebar:
    st.image("Nakshatra_transparent_1.png", use_container_width=True)
    st.markdown("### Spectral Intelligence Platform")
    st.markdown(f"**User:** {st.session_state.user['email']}")
    if st.button("Logout"): st.session_state.user=None; st.rerun()
    st.divider()

    input_source = st.radio("Data Source", ["Upload File","Live Camera","Simulation"])

    if st.session_state.data_ready:
        with st.expander("Calibration Settings"):
            st.session_state.start_wl = st.number_input(
                "Start Å", 
                value=st.session_state.get("start_wl", DEFAULT_START_WL)
            )
            st.session_state.disp = st.number_input(
                "Å / pixel", 
                value=st.session_state.get("disp", DEFAULT_DISP)
            )


# ================= MAIN CONTENT =================
st.title("Spectral Analysis Dashboard")

data=None

if input_source=="Upload File":
    f=st.file_uploader("Upload FIT/FITS/SER",["fit","fits"])
    if f:
        with fits.open(f) as h: data=h[0].data

elif input_source=="Live Camera":
    img=st.camera_input("Capture")
    if img:
        frame=np.array(Image.open(img).convert("L"))
        data=frame

else:
    x=np.linspace(4000,7000,1000)
    data=100+(x-4000)*.03+500*np.exp(-.5*((x-6563)/10)**2)

if data is not None:
    st.session_state.data_ready=True

    flux = np.mean(data,axis=0) if data.ndim==2 else data
    if 'smooth' in locals() and smooth>1:
        flux=savgol_filter(flux,smooth if smooth%2 else smooth+1,3,deriv)

    # ---- Safe calibration values ----
    start_wl = st.session_state.get("start_wl", DEFAULT_START_WL)
    disp = st.session_state.get("disp", DEFAULT_DISP)
    
    x_axis = start_wl + (np.arange(len(flux)) * disp)


    fig=go.Figure(go.Scatter(x=x_axis,y=flux,fill='tozeroy',line=dict(color="#4FC3F7")))
    fig.update_layout(template="plotly_dark",height=350,margin=dict(l=10,r=10,t=10,b=10))
    st.plotly_chart(fig,use_container_width=True)

    tab=st.radio("Mode",["Analyze","ML Lab","Library"],horizontal=True,label_visibility="collapsed")

    if tab=="Analyze":
        st.success("Spectrum loaded successfully")
        if st.button("Save Analysis to Library"):
            if "saved_spectra" not in st.session_state:
                st.session_state.saved_spectra = []
        
            st.session_state.saved_spectra.append({
                "name": f"Spectrum {len(st.session_state.saved_spectra)+1}",
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                "wavelength": x_axis.tolist(),
                "flux": flux.tolist()
            })
        
            st.toast("Saved to Library")


    if tab=="ML Lab":
        lbl=st.text_input("Label")
        if st.button("Add Sample"):
            _,v=resample(x_axis,normalize(flux))
            row=pd.DataFrame([[lbl]+list(v)],columns=st.session_state.dataset.columns)
            st.session_state.dataset=pd.concat([st.session_state.dataset,row])

        if st.button("Train"):
            df=st.session_state.dataset
            if len(df)>1:
                X,y=df.drop("label",1),df["label"]
                clf=RandomForestClassifier()
                clf.fit(X,y)
                st.session_state.trained_model=clf
                st.success("Model trained")

    if tab == "Library":
        st.subheader("Saved Analyses")
    
        if "saved_spectra" not in st.session_state:
            st.session_state.saved_spectra = []
    
        if len(st.session_state.saved_spectra) == 0:
            st.info("No saved analyses yet.")
        else:
            for i, item in enumerate(st.session_state.saved_spectra):
                st.markdown(
                    f"**{i+1}.** {item['name']} — {item['timestamp']}"
                )

