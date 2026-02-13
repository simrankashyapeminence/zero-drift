import streamlit as st
import pandas as pd
import httpx
import base64
import os
import re
import time
import asyncio
from PIL import Image
from io import BytesIO

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Zero-Drift AI Product Studio",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ENV SYNC (Streamlit Secrets -> OS Environ) ---
# This allows the backend service (using pydantic settings) to see the secrets
try:
    if "NANO_BANANA_API_KEY" in st.secrets:
        os.environ["NANO_BANANA_API_KEY"] = st.secrets["NANO_BANANA_API_KEY"]
    if "NANO_BANANA_BASE_URL" in st.secrets:
        os.environ["NANO_BANANA_BASE_URL"] = st.secrets["NANO_BANANA_BASE_URL"]
    if "GEMINI_MODEL_VERSION" in st.secrets:
        os.environ["GEMINI_MODEL_VERSION"] = st.secrets["GEMINI_MODEL_VERSION"]
except:
    pass # Running locally or no secrets configured

# --- STYLING ---
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stButton>button {
        width: 100%; border-radius: 8px; height: 3em;
        background: linear-gradient(135deg, #ff4b4b, #ff6b6b);
        color: white; font-weight: bold; border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #ff6b6b, #ff4b4b);
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(255, 75, 75, 0.4);
    }
    .stMetric { background-color: #1e2130; padding: 15px; border-radius: 10px; }
    h1 { background: linear-gradient(90deg, #ff4b4b, #ff8c42); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
</style>
""", unsafe_allow_html=True)

# --- GEMINI CONFIG (from secrets or env) ---
def get_api_key():
    """Get API key from Streamlit secrets or environment."""
    try:
        return st.secrets["NANO_BANANA_API_KEY"]
    except:
        return os.getenv("NANO_BANANA_API_KEY", "")

def get_base_url():
    try:
        return st.secrets.get("NANO_BANANA_BASE_URL", "https://generativelanguage.googleapis.com/v1beta")
    except:
        return os.getenv("NANO_BANANA_BASE_URL", "https://generativelanguage.googleapis.com/v1beta")

def get_model_version():
    try:
        return st.secrets.get("GEMINI_MODEL_VERSION", "gemini-2.0-flash-exp-image-generation")
    except:
        return os.getenv("GEMINI_MODEL_VERSION", "gemini-2.0-flash-exp-image-generation")

# --- EXCEL PARSING ---
def parse_excel(file_bytes):
    """Parse Excel and return metadata list."""
    df = pd.read_excel(BytesIO(file_bytes))
    df.columns = [str(col).lower().strip().replace(' ', '_') for col in df.columns]
    
    column_mapping = {
        'product_code': ['product_code', 'code', 'p_code', 'sku', 'nazwa_bazowa', 'nr_artykułu', 'id'],
        'product_name': ['product_name', 'name', 'title', 'nazwa_handlowa', 'opis'],
        'product_type': ['product_type', 'type', 'category', 'type_of_product', 'rodzaj'],
        'gender': ['płeć', 'gender', 'sex'],
        'sport': ['sport_dominujący', 'sport', 'discipline'],
        'pose': ['pose', 'action', 'position', 'poza'],
        'environment': ['environment', 'background', 'location', 'scene', 'otoczenie']
    }
    
    results = []
    for _, row in df.iterrows():
        metadata = {}
        row_values = [str(v).strip().upper() for v in row.values]
        
        for key, variations in column_mapping.items():
            for var in variations:
                if var in df.columns:
                    metadata[key] = str(row[var])
                    break
            if key not in metadata:
                metadata[key] = "N/A"
        
        metadata['raw_row_values'] = row_values
        results.append(metadata)
    
    return results

def deep_clean(text):
    return re.sub(r'[^A-Z0-9]', '', str(text).upper())

def match_images_to_metadata(metadata_list, image_files):
    """Match uploaded images to Excel metadata."""
    mapped = []
    
    for img_name, img_bytes in image_files:
        img_base = os.path.splitext(img_name)[0]
        clean_img = deep_clean(img_base)
        matched = False
        
        for row in metadata_list:
            p_code = row.get('product_code', '')
            p_name = row.get('product_name', '')
            all_vals = [p_code, p_name] + row.get('raw_row_values', [])
            
            for raw_val in all_vals:
                if not raw_val or str(raw_val).upper() == "N/A":
                    continue
                clean_val = deep_clean(raw_val)
                if len(clean_val) < 2:
                    continue
                
                if clean_val == clean_img or clean_img.startswith(clean_val) or clean_val.startswith(clean_img):
                    mapped.append({
                        'product_code': str(p_code) if p_code != "N/A" else clean_val,
                        'product_name': str(row.get('product_name', 'Product')),
                        'product_type': str(row.get('product_type', 'Fashion')),
                        'gender': str(row.get('gender', 'N/A')),
                        'sport': str(row.get('sport', 'N/A')),
                        'pose': str(row.get('pose', 'N/A')),
                        'environment': str(row.get('environment', 'N/A')),
                        'image_name': img_name,
                        'image_bytes': img_bytes
                    })
                    matched = True
                    break
            if matched:
                break
        
        # Fallback: prefix matching
        if not matched:
            parts = re.split(r'[-_ ]+', img_base)
            if parts:
                prefix = deep_clean(parts[0])
                if len(prefix) >= 2:
                    for row in metadata_list:
                        for raw_val in [row.get('product_code', '')] + row.get('raw_row_values', []):
                            if deep_clean(raw_val) == prefix:
                                mapped.append({
                                    'product_code': str(raw_val),
                                    'product_name': str(row.get('product_name', 'Product')),
                                    'product_type': str(row.get('product_type', 'Fashion')),
                                    'gender': str(row.get('gender', 'N/A')),
                                    'sport': str(row.get('sport', 'N/A')),
                                    'pose': str(row.get('pose', 'N/A')),
                                    'environment': str(row.get('environment', 'N/A')),
                                    'image_name': img_name,
                                    'image_bytes': img_bytes
                                })
                                matched = True
                                break
                    if matched:
                        break
    
    return mapped

# --- GEMINI IMAGE GENERATION ---
# --- GEMINI IMAGE GENERATION (VIA BACKEND SERVICE) ---
async def generate_image(product_info_dict, image_data_list):
    """
    Call the backend NanoBananaService to generate the image.
    This ensures we use the EXACT SAME prompt logic as the API.
    """
    import tempfile
    import shutil
    from app.services.nano_banana_service import NanoBananaService
    from app.models.metadata import ProductMetadata
    from app.core.config import settings
    
    # 1. Initialize Service
    service = NanoBananaService()
    
    # 2. Create ProductMetadata object
    try:
        product = ProductMetadata(
            product_code=product_info_dict.get('product_code', 'UNKNOWN'),
            product_name=product_info_dict.get('product_name', 'Product'),
            product_type=product_info_dict.get('product_type', 'Fashion'),
            gender=product_info_dict.get('gender', 'Unisex'),
            sport=product_info_dict.get('sport', 'General'),
            pose=product_info_dict.get('pose', 'N/A'),
            environment=product_info_dict.get('environment', 'N/A'),
            image_filename="temp.png" # Placeholder
        )
    except Exception as e:
        return None, f"Metadata validation failed: {e}"

    # 3. Save uploaded images to temp files
    temp_dir = tempfile.mkdtemp()
    image_paths = []
    
    try:
        for idx, (img_bytes, img_name) in enumerate(image_data_list):
            # Clean filename
            safe_name = "".join([c for c in img_name if c.isalnum() or c in ('-', '_', '.')])
            temp_path = os.path.join(temp_dir, safe_name)
            with open(temp_path, "wb") as f:
                f.write(img_bytes)
            image_paths.append(temp_path)
        
        # 4. Call Service
        # We use generate_tryon_image for single items
        # Ensure we're in an async context
        try:
            output_path = await service.generate_tryon_image(
                product=product,
                image_paths=image_paths,
                variation_index=0
            )
            
            if output_path and os.path.exists(output_path):
                with open(output_path, "rb") as f:
                    result_bytes = f.read()
                return result_bytes, None
            else:
                return None, "Service returned complete status but no output file found."
                
        except Exception as e:
            return None, f"Service Generation Error: {str(e)}"

    except Exception as e:
        return None, f"Temp file error: {str(e)}"
    finally:
        # Cleanup temp upload files
        shutil.rmtree(temp_dir, ignore_errors=True)

# --- SESSION STATE ---
if "results" not in st.session_state:
    st.session_state.results = []
if "processing" not in st.session_state:
    st.session_state.processing = False

# --- UI ---
st.title("🎨 Zero-Drift AI Product Studio")
st.markdown("### High-Fidelity Athlete Image Generation")

# Check API Key
api_key = get_api_key()
if not api_key:
    st.warning("⚠️ **API Key Required!** Add `NANO_BANANA_API_KEY` in the app's **Settings → Secrets** section.")
    st.code("""
# In Streamlit Cloud: Settings → Secrets
NANO_BANANA_API_KEY = "your-api-key-here"
NANO_BANANA_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
GEMINI_MODEL_VERSION = "gemini-2.0-flash-exp-image-generation"
    """, language="toml")

# --- SIDEBAR ---
with st.sidebar:
    st.header("📤 Upload Assets")
    excel_file = st.file_uploader("Upload Product Excel (.xlsx)", type=["xlsx"])
    
    st.write("---")
    st.subheader("📸 Product Images")
    st.info("Upload product images. They will be matched to the Excel data automatically.")
    uploaded_images = st.file_uploader("Select Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if st.button("🔄 Reset Studio"):
        st.session_state.results = []
        st.session_state.processing = False
        st.rerun()

    start_btn = st.button("🚀 Start Processing", disabled=st.session_state.processing)

# --- PROCESSING ---
if start_btn and excel_file and uploaded_images and api_key:
    st.session_state.processing = True
    st.session_state.results = []
    
    # 1. Parse Excel
    with st.spinner("📊 Parsing Excel metadata..."):
        metadata_list = parse_excel(excel_file.getvalue())
        st.success(f"✅ Parsed {len(metadata_list)} rows from Excel")
    
    # 2. Match images
    image_files = [(img.name, img.getvalue()) for img in uploaded_images]
    mapped = match_images_to_metadata(metadata_list, image_files)
    
    if not mapped:
        st.error("❌ No images matched Excel data. Check that image names contain product codes.")
        st.session_state.processing = False
        st.stop()
    
    st.success(f"🎯 Matched {len(mapped)} images to products")
    
    # 3. Group by product code
    grouped = {}
    for item in mapped:
        code = item['product_code']
        if code not in grouped:
            grouped[code] = {'info': item, 'images': []}
        grouped[code]['images'].append((item['image_bytes'], item['image_name']))
    
    total_products = len(grouped)
    st.info(f"🔄 Generating **{total_products}** product(s) using **{len(mapped)}** reference images...")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    results_container = st.container()
    
    # 4. Generate for each product
    # 4. Aggregate products for SMART BATCH PROCESSING
    # Instead of looping one-by-one, we send ALL products to the backend.
    # The backend will decide: Same Sport? -> Combine into Outfit. Different? -> Separate.
    
    status_text.info("📦 Preparing batch for AI processing...")
    
    # We need to instantiate the service here to use batch_process
    # Note: Environment variables for API key are already set in top of gui.py
    from app.services.nano_banana_service import NanoBananaService
    from app.models.metadata import ProductMetadata
    
    service = NanoBananaService()
    batch_products = []
    
    # Helper to save temp files and create metadata objects
    temp_files_to_cleanup = []
    
    try:
        current_time = int(time.time())
        temp_dir = f"temp_upload_{current_time}"
        os.makedirs(temp_dir, exist_ok=True)
        
        for code, data in grouped.items():
            product_info = data['info']
            image_list = data['images'] # List of (bytes, name)
            
            # Save first image as primary reference (or handle multiple in service)
            # Service expects ONE image_filename per product metadata currently?
            # NanoBananaService.generate_tryon_image takes `image_paths` list.
            # But ProductMetadata has `image_filename`.
            # Let's save all images, but metadata only holds primary.
            # Actually, batch_process uses `p.image_filename`.
            
            # We'll save the first image as the primary one for the metadata
            primary_image_path = ""
            saved_paths = []
            
            for idx, (img_bytes, img_name) in enumerate(image_list):
                # Clean filename
                clean_name = f"{code}_{idx}_{int(time.time())}.png"
                save_path = os.path.join(temp_dir, clean_name)
                with open(save_path, "wb") as f:
                    f.write(img_bytes)
                saved_paths.append(save_path)
                
                # We also need to copy/link to settings.IMAGES_DIR because service looks there?
                # Service code: image_paths = [os.path.join(settings.IMAGES_DIR, p.image_filename) for p in group]
                # We need to ensure settings.IMAGES_DIR (uploads/images) has these files OR hack it.
                # Service uses os.path.join, so if filename is absolute path, it might fail or work?
                # os.path.join("dir", "/abs/path") -> "/abs/path" (on Linux).
                # So we can put absolute path in metadata.image_filename!
                
                if idx == 0:
                    primary_image_path = os.path.abspath(save_path)
            
            # Create Metadata Object for EACH saved image to maximize accuracy
            # Backend will re-group them by product code.
            for path in saved_paths:
                meta = ProductMetadata(
                    product_code=str(product_info.get('product_code', 'N/A')),
                    product_name=str(product_info.get('product_name', 'N/A')),
                    product_type=str(product_info.get('product_type', 'N/A')),
                    gender=str(product_info.get('gender', 'N/A')),
                    sport=str(product_info.get('sport', 'N/A')),
                    pose=str(product_info.get('pose', 'N/A')),
                    environment=str(product_info.get('environment', 'N/A')),
                    image_filename=os.path.abspath(path) 
                )
                batch_products.append(meta)
            
            temp_files_to_cleanup.append(temp_dir) # directory to clean later
            
        # CALL BATCH PROCESS
        # Log unique products count instead of total metadata count to avoid confusion
        status_text.info(f"🚀 AI Processing started for {len(grouped)} unique product(s) using {len(batch_products)} reference images...")
        loop = asyncio.new_event_loop()
        generated_paths = loop.run_until_complete(service.batch_process(batch_products))
        loop.close()
        
        # Display Results
        progress_bar.progress(1.0)
        
        if not generated_paths:
            st.warning("⚠️ No images were generated. Check logs.")
        
        for path in generated_paths:
            if os.path.exists(path):
                file_name = os.path.basename(path)
                with open(path, "rb") as f:
                    img_bytes = f.read()
                
                # Add to history
                st.session_state.results.append({
                    'code': file_name, # Use filename as code since it might be outfit
                    'name': "AI Generated",
                    'image_bytes': img_bytes
                })
                
                with results_container:
                     st.image(img_bytes, caption=file_name, use_container_width=True)
                     st.download_button(
                        label=f"📥 Download {file_name}",
                        data=img_bytes,
                        file_name=file_name,
                        mime="image/png"
                     )
            else:
                st.error(f"Generated file not found: {path}")

    except Exception as e:
        st.error(f"Pipeline Error: {str(e)}")
        # Cleanup can happen here logic
    
    # Cleanup temp
    # shutil.rmtree(temp_dir) if needed
    status_text.success("✅ Batch Processing Complete!")
    
    # Done
    progress_bar.progress(1.0)
    status_text.empty()
    st.balloons()
    st.success(f"🎉 Processing Complete! Generated {len(st.session_state.results)}/{total_products} images.")
    st.session_state.processing = False

elif start_btn and not api_key:
    st.error("❌ Please configure your API key first (see instructions above).")

elif start_btn and (not excel_file or not uploaded_images):
    st.warning("⚠️ Please upload both an Excel file and at least one image.")

# --- SHOW PREVIOUS RESULTS ---
if st.session_state.results and not st.session_state.processing:
    st.write("---")
    st.subheader("✨ Generated Results")
    cols = st.columns(2)
    for idx, result in enumerate(st.session_state.results):
        col = cols[idx % 2]
        with col:
            img = Image.open(BytesIO(result['image_bytes']))
            st.image(img, use_container_width=True, caption=f"{result['code']} - {result['name']}")
            st.download_button(
                label=f"📥 Download {result['code']}",
                data=result['image_bytes'],
                file_name=f"zero_drift_{result['code']}.png",
                mime="image/png",
                key=f"dl_prev_{idx}"
            )

# --- HOME STATE ---
if not st.session_state.results and not st.session_state.processing:
    st.write("### 📋 Instructions")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### 1. Prepare Excel")
        st.write("Ensure your Excel has columns like `product_code`, `product_name`, `gender`, `sport`, `pose`, and `environment`.")
    with col2:
        st.markdown("#### 2. Name Images")
        st.write("Images should be named after the `product_code` for automatic matching (e.g., `RSO6-EUW-1.png`).")
    with col3:
        st.markdown("#### 3. Upload & Run")
        st.write("Upload both files in the sidebar and click **Start Processing**.")

st.markdown("---")
st.caption("Powered by Gemini AI | Zero-Drift Strategy | Built with ❤️")
