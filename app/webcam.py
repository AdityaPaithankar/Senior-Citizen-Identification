
import csv
import time
from datetime import datetime
from pathlib import Path

import cv2

from pipeline import SeniorCitizenPipeline


BASE_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = LOG_DIR / "visitor_log.csv"


def initialize_csv():
    if not CSV_PATH.exists():
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([
                "person_id",
                "estimated_age",
                "predicted_gender",
                "visit_time",
                "senior_status"
            ])


def log_visitor(person_id, age, gender, senior):
    initialize_csv()

    with open(CSV_PATH, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            person_id,
            age,
            gender,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Senior Citizen" if senior else "Non-Senior"
        ])


def main():
    pipeline = SeniorCitizenPipeline()
    initialize_csv()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Webcam could not be opened.")
        return

    next_id = 1
    last_logged = {}
    last_prediction_time = 0
    cached_results = []

    print("Press Q to quit.")

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("Could not read webcam frame.")
                break

            # Run predictions periodically to reduce computation.
            if time.time() - last_prediction_time >= 1.0:
                cached_results = pipeline.predict_faces(frame)
                last_prediction_time = time.time()

            for person in cached_results:
                x, y, w, h = person["box"]
                age = person["age"]
                gender = person["gender"]
                senior = person["senior"]

                # Temporary ID for this detection.
                # Persistent tracking is added in the next step.
                person_id = f"P{next_id}"

                if person_id not in last_logged:
                    log_visitor(person_id, age, gender, senior)
                    last_logged[person_id] = time.time()
                    next_id += 1

                status = "Senior Citizen" if senior else "Non-Senior"
                color = (0, 165, 255) if senior else (0, 200, 0)

                cv2.rectangle(
                    frame, (x, y), (x + w, y + h), color, 2
                )

                label = f"{age} | {gender} | {status}"

                cv2.putText(
                    frame,
                    label,
                    (x, max(20, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2
                )

            cv2.imshow("Senior Citizen Identification", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
