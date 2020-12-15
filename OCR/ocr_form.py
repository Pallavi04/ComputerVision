import alignDocument
from collections import namedtuple
import pytesseract
import argparse
import imutils
import cv2

def cleanup_text(text):
	return "".join([c if ord(c) < 128 else "" for c in text]).strip()

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image that we'll align to template")
ap.add_argument("-t", "--template", required=True,
	help="path to input template image")
args = vars(ap.parse_args())

OCRLocation = namedtuple("OCRLocation", ["id", "bbox",
	"filter_keywords"])
# define the locations of each area of the document we wish to OCR
OCR_LOCATIONS = [
	OCRLocation("SSN Number", (308, 86, 271, 33),
		["ssnNumber"]),
	OCRLocation("Employee ID", (68, 134, 613, 34),
		["EmployeeID"]),
	OCRLocation("Employee Address", (68, 383, 613, 187),
		["employee", "name", "address"]),
	OCRLocation("Employers Address", (68, 186, 613, 134),
		["employee", "name", "address"]),
	OCRLocation("State", (68, 607, 61, 34),
		["State"]),
	OCRLocation("Emp State ID", (129, 608, 269, 35),
		["stateId"])
]

print("[INFO] loading images...")
image = cv2.imread("Input.jpg")
template = cv2.imread("W2.jpg")
print("[INFO] aligning images...")
aligned = alignDocument.align_image(image, template)

print("[INFO] OCR'ing document...")
parsingResults = []
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\palla\AppData\Local\Tesseract-OCR\tesseract.exe'
for loc in OCR_LOCATIONS:
	# extract the OCR ROI from the aligned image
	(x, y, w, h) = loc.bbox
	roi = aligned[y:y + h, x:x + w]
	rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
	text = pytesseract.image_to_string(rgb)
	for line in text.split("\n"):
		if len(line) == 0:
			continue
		lower = line.lower()
		count = sum([lower.count(x) for x in loc.filter_keywords])
		if count == 0:
			parsingResults.append((loc, line))

results = {}
for (loc, line) in parsingResults:
	r = results.get(loc.id, None)
	if r is None:
		results[loc.id] = (line, loc._asdict())
	else:
		(existingText, loc) = r
		text = "{}\n{}".format(existingText, line)
		results[loc["id"]] = (text, loc)

for (locID, result) in results.items():
	(text, loc) = result
	print(loc["id"])
	print("=" * len(loc["id"]))
	print("{}\n\n".format(text))
	(x, y, w, h) = loc["bbox"]
	clean = cleanup_text(text)
	cv2.rectangle(aligned, (x, y), (x + w, y + h), (0, 255, 0), 2)
	for (i, line) in enumerate(text.split("\n")):
		startY = y + (i * 70) + 40
		cv2.putText(aligned, line, (x, startY),
			cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 255), 5)

# show the input and output images, resizing it such that they fit on screen
cv2.imshow("Input", imutils.resize(image, width=700))
cv2.imshow("Output", imutils.resize(aligned, width=700))
cv2.waitKey(0)