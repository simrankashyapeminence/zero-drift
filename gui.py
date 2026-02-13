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
    for i, (code, data) in enumerate(grouped.items()):
        status_text.info(f"⚙️ Generating **{code}** ({i+1}/{total_products})...")
        
        # Run async generation
        loop = asyncio.new_event_loop()
        img_bytes, error = loop.run_until_complete(
            generate_image(data['info'], data['images'])
        )
        loop.close()
        
        if img_bytes:
            st.session_state.results.append({
                'code': code,
                'name': data['info']['product_name'],
                'image_bytes': img_bytes
            })
            progress_bar.progress((i + 1) / total_products)
            
            # Show result immediately
            with results_container:
                img = Image.open(BytesIO(img_bytes))
                cols = st.columns([2, 1])
                with cols[0]:
                    st.image(img, use_container_width=True, caption=f"{code} - {data['info']['product_name']}")
                with cols[1]:
                    st.download_button(
                        label=f"📥 Download {code}",
                        data=img_bytes,
                        file_name=f"zero_drift_{code}.png",
                        mime="image/png",
                        key=f"dl_{code}"
                    )
        else:
            st.warning(f"⚠️ Failed for {code}: {error}")
        
        # Small delay between requests
        if i < total_products - 1:
            time.sleep(2)
    
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
