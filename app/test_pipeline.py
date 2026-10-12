
from pathlib import Path
import sys
import cv2

APP_DIR = Path.cwd() / "app"

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from pipeline import SeniorCitizenPipeline

pipeline = SeniorCitizenPipeline()

image_path = input("Enter the path to a test image: ").strip().strip('"')
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(f"Could not read image: {image_path}")

results = pipeline.predict_faces(image)

for person_id, person in enumerate(results, start=1):
    x, y, w, h = person["box"]

    status = "Senior Citizen" if person["senior"] else "Non-Senior"
    color = (0, 165, 255) if person["senior"] else (0, 200, 0)

    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

    label = (
        f'ID {person_id} | {person["age"]} | '
        f'{person["gender"]} | {status}'
    )

    cv2.putText(
        image, label, (x, max(20, y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
    )

print("Faces detected:", len(results))

cv2.imshow("Senior Citizen Pipeline Test", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
