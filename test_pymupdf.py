import pymupdf
import os

pdf_path = os.path.join(
    "data",
    "01_Weston_2014_Memory_Networks.pdf"
)

print("Opening:", pdf_path)

doc = pymupdf.open(pdf_path)

print("Pages:", len(doc))

page = doc[0]

text = page.get_text("text")

print("\n===== EXTRACTED TEXT =====")
print(text[:3000])

doc.close()