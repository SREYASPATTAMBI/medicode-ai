from ocr_agent import read_handwriting

# test the OCR agent
result = read_handwriting("../data/test.jpg")
if result:
    print("\n🎉 OCR Agent is working!")
    print(f"Extracted text: {result}")
else:
    print("❌ Something went wrong")