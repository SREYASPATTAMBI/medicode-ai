from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

# load the model once when the file is imported
print("🔄 Loading TrOCR handwriting model...")
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
print("✅ OCR model loaded!")

def read_handwriting(image_path):
    """
    Takes a path to an image of handwritten text
    Returns the extracted text as a string
    """
    try:
        # open the image
        image = Image.open(image_path).convert("RGB")
        
        # prepare image for the model
        pixel_values = processor(image, return_tensors="pt").pixel_values
        
        # generate text from image
        generated_ids = model.generate(pixel_values)
        
        # decode the output to readable text
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        print(f"✅ OCR complete!")
        print(f"📝 Extracted text: {text}")
        
        return text
    
    except Exception as e:
        print(f"❌ OCR failed: {e}")
        return None