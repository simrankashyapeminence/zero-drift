import streamlit as st
import requests
import time
import os
from PIL import Image
import pandas as pd
from io import BytesIO

# --- CONFIGURATION ---
API_BASE_URL = "http://localhost:8080"
UPLOAD_URL = f"{API_BASE_URL}/api/v1/processing/upload"
STATUS_URL = f"{API_BASE_URL}/api/v1/processing/status"

st.set_page_config(
    page_title="Zero-Drift AI Product Studio",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- STYLING ---
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
    }
    .stMetric {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE ---
if "job_id" not in st.session_state:
    st.session_state.job_id = None
if "processing" not in st.session_state:
    st.session_state.processing = False

def check_api_health():
    try:
        requests.get(f"{API_BASE_URL}/")
        return True
    except:
        return False

# --- UI CONTENT ---
st.title("🎨 Zero-Drift AI Product Studio")
st.markdown("### High-Fidelity Athlete Image Generation")

if not check_api_health():
    st.error(f"❌ Backend API not detected on {API_BASE_URL}. Please ensure the FastAPI server is running.")
    st.stop()

# --- SIDEBAR: Uploads ---
with st.sidebar:
    st.header("📤 Upload Assets")
    excel_file = st.file_uploader("Upload Product Excel (.xlsx)", type=["xlsx"])
    
    st.write("---")
    st.subheader("📸 Product Images")
    st.info("Upload standard product images. They will be matched to the Excel data automatically.")
    uploaded_images = st.file_uploader("Select Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if st.sidebar.button("🔄 Reset Studio"):
        st.session_state.job_id = None
        st.session_state.processing = False
        st.rerun()

    if st.button("🚀 Start Processing") and excel_file and uploaded_images:
        st.session_state.processing = True
        st.session_state.job_id = None
        
        with st.spinner("Uploading and starting job..."):
            try:
                files = [
                    ("excel_file", (excel_file.name, excel_file.getvalue(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"))
                ]
                
                for img in uploaded_images:
                    files.append(("images", (img.name, img.getvalue(), img.type)))
                
                response = requests.post(UPLOAD_URL, files=files)
                
                if response.status_code == 200:
                    data = response.json()
                    st.session_state.job_id = data.get("job_id")
                    st.success(f"✅ Job Started! ID: {st.session_state.job_id}")
                else:
                    st.error(f"❌ Upload failed: {response.text}")
                    st.session_state.processing = False
            except Exception as e:
                st.error(f"❌ Connection error: {str(e)}")
                st.session_state.processing = False

# --- MAIN AREA ---
if st.session_state.job_id:
    st.write("---")
    status_container = st.empty()
    progress_bar = st.progress(0)
    
    # Simple Polling Loop
    max_retries = 100
    for _ in range(max_retries):
        try:
            resp = requests.get(f"{STATUS_URL}/{st.session_state.job_id}")
            if resp.status_code == 200:
                job_data = resp.json()
                status = job_data.get("status")
                
                # Update progress
                total = job_data.get("total_items", 1)
                completed = job_data.get("completed_items", 0)
                progress = min(completed / total if total > 0 else 0, 1.0)
                progress_bar.progress(progress)
                
                status_container.info(f"**Status:** {status.upper()} | **Progress:** {completed}/{total}")
                
                if job_data.get("results"):
                    st.subheader("✨ Generated Results")
                    results = job_data["results"]
                    cols = st.columns(2)
                    for idx, img_path in enumerate(results):
                        col = cols[idx % 2]
                        if os.path.exists(img_path):
                            img = Image.open(img_path)
                            col.image(img, use_container_width=True, caption=os.path.basename(img_path))
                            
                            with open(img_path, "rb") as file:
                                col.download_button(
                                    label=f"📥 Download {idx+1}",
                                    data=file,
                                    file_name=os.path.basename(img_path),
                                    mime="image/png",
                                    key=f"dl_{idx}"
                                )

                if status == "completed":
                    st.balloons()
                    st.success("🎉 Processing Complete!")
                    st.session_state.processing = False # Allow new uploads
                    
                    if st.button("🆕 Process Another Batch"):
                        st.session_state.job_id = None
                        st.rerun()
                    break
                elif status == "failed":
                    st.error(f"❌ Job Failed: {job_data.get('error', 'Unknown error')}")
                    st.session_state.processing = False
                    if st.button("Retry"):
                        st.session_state.job_id = None
                        st.rerun()
                    break
            else:
                st.error("Failed to fetch status.")
                st.session_state.processing = False
                break
        except Exception as e:
            st.error(f"Error polling status: {e}")
            st.session_state.processing = False
            break
        
        time.sleep(5)
else:
    # Home State
    if not st.session_state.processing:
        st.write("### Instructions")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("#### 1. Prepare Excel")
            st.write("Ensure your Excel has columns like `product_code`, `product_name`, `gender`, and `sport`.")
        with col2:
            st.markdown("#### 2. Name Images")
            st.write("Images should be named after the `product_code` or `product_name` for automatic matching.")
        with col3:
            st.markdown("#### 3. Upload & Run")
            st.write("Upload both file types in the sidebar and click **Start Processing**.")
        
        st.image("https://img.freepik.com/free-vector/fashion-designer-concept-illustration_114360-1049.jpg", width=400)
    else:
        st.info("Waiting for job initialization...")

st.markdown("---")
st.caption("Powered by Nano Banana AI | Zero-Drift Strategy")
