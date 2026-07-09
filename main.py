import sys

from src.ocr_engine_rapid import RapidOCREngine

# this is tester for the ocr engine
engine = RapidOCREngine()

if len(sys.argv) > 1:
    first_arg = sys.argv[1]
else:
    print("No arguments provided!")


result = engine.extract_text(first_arg)
print(result)
