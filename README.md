# Zero-Drift Image Generation Service (Nano Banana API)

This is a production-grade FastAPI service designed to automate virtual try-on image generation while preserving garment details (Zero-Drift).

## Architecture
- **Framework**: FastAPI (MVC Pattern)
- **Logging**: Loguru (Console + File rotation)
- **Data Processing**: Pandas / Openpyxl (Excel metadata)
- **AI Integration**: Nano Banana API (Gemini-powered image synthesis)
- **Background Tasks**: FastAPI BackgroundTasks for batch processing

## Features
- **Zero-Drift**: Advanced prompting and API parameters ensure the product image remains identical (geometry, texture, seams) in the generated result.
- **Excel Mapping**: Automatically maps product codes from Excel to uploaded image filenames.
- **Detailed Logging**: Every step of the pipeline is logged for transparency and debugging.
- **Production Ready**: Structured configuration, environment variables, and modular services.

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Configure `.env`:
   - Add your `NANO_BANANA_API_KEY`.
3. Start the server:
   ```bash
   python -m app.main
   ```

## API Usage Flow (No UI)

### 1. Upload Excel Sheet
`POST /api/v1/upload/excel`
- Body: `file` (Excel file with columns: Product code, product name, type of product)

### 2. Upload Product Images
`POST /api/v1/upload/images`
- Body: `files` (Multiple product image files. Filenames should ideally match the product codes in your Excel).

### 3. Trigger Processing
`POST /api/v1/processing/process`
- Body (JSON):
  ```json
  {
    "excel_filename": "products.xlsx",
    "image_filenames": ["sku001.jpg", "sku002.png"]
  }
  ```
- **Response**: Confirms the process has started. The generation runs in the background.

## Results
- Generated images are saved in the `exports/` folder.
- Detailed logs are available in `logs/app.log`.
