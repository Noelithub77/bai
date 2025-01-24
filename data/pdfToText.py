import fitz  # PyMuPDF
import pdfplumber
import pytesseract
from PIL import Image
import io
import os
import json

def extract_pdf_to_text(pdf_path, output_txt_path, ocr_fallback=False):
    """
    Extracts text, tables, images (with OCR), and metadata from a PDF.
    Args:
        pdf_path (str): Path to the input PDF file.
        output_txt_path (str): Path to save the extracted text.
        ocr_fallback (bool): Use OCR for image-based/scanned PDFs.
    """
    doc = fitz.open(pdf_path)
    full_text = []
    metadata = doc.metadata
    extracted_tables = []
    extracted_images = []

    # Extract metadata
    full_text.append(f"=== METADATA ===\n{json.dumps(metadata, indent=2)}\n")

    # Extract text, tables, and images
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        page_text = page.get_text()
        full_text.append(f"\n=== PAGE {page_num + 1} TEXT ===\n{page_text}")

        # Extract tables using pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            pdf_page = pdf.pages[page_num]
            tables = pdf_page.extract_tables()
            if tables:
                extracted_tables.append(f"\n=== PAGE {page_num + 1} TABLES ===")
                for table in tables:
                    extracted_tables.append("\n".join(["\t".join(row) for row in table]))

        # Extract images and apply OCR if needed
        img_list = page.get_images(full=True)
        for img_index, img in enumerate(img_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            img = Image.open(io.BytesIO(image_bytes))
            img_path = f"page_{page_num}_img_{img_index}.png"
            img.save(img_path)
            if ocr_fallback:
                ocr_text = pytesseract.image_to_string(img)
                extracted_images.append(f"\n=== PAGE {page_num + 1} IMAGE {img_index} OCR TEXT ===\n{ocr_text}")

    # Combine all content
    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(full_text))
        f.write("\n".join(extracted_tables))
        f.write("\n".join(extracted_images))

    # Clean up temporary images
    for img_file in os.listdir():
        if img_file.startswith("page_") and img_file.endswith(".png"):
            os.remove(img_file)

# Usage
input_pdf_path = "dnf.pdf"
output_txt_path = os.path.splitext(input_pdf_path)[0] + ".txt"
extract_pdf_to_text(input_pdf_path, output_txt_path)