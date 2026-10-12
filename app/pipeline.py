
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "best_age_gender_model.keras"

IMG_SIZE = 128

# Model's estimated-age cutoff, tuned on validation data.
# The actual ground-truth definition remains age > 60.
SENIOR_PREDICTION_THRESHOLD = 50


class SeniorCitizenPipeline:
    def __init__(self, model_path=MODEL_PATH):
        self.model = tf.keras.models.load_model(model_path)

        cascade_path = (
            cv2.data.haarcascades
            + "haarcascade_frontalface_default.xml"
        )

        self.face_detector = cv2.CascadeClassifier(cascade_path)

        if self.face_detector.empty():
            raise RuntimeError("Could not load face detector.")

    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40)
        )

        return faces

    def predict_faces(self, frame):
        faces = self.detect_faces(frame)

        if len(faces) == 0:
            return []

        face_images = []

        for x, y, w, h in faces:
            # Slightly expand crop around the face.
            pad_x = int(w * 0.15)
            pad_y = int(h * 0.15)

            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(frame.shape[1], x + w + pad_x)
            y2 = min(frame.shape[0], y + h + pad_y)

            crop = frame[y1:y2, x1:x2]
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
            crop = crop.astype(np.float32) / 255.0

            face_images.append(crop)

        batch = np.asarray(face_images, dtype=np.float32)

        # Model has two named outputs: age and gender.
        predictions = self.model.predict(batch, verbose=0)

        if isinstance(predictions, dict):
            ages = np.asarray(predictions["age"]).reshape(-1)
            genders = np.asarray(predictions["gender"]).reshape(-1)
        else:
            output_map = dict(zip(self.model.output_names, predictions))
            ages = np.asarray(output_map["age"]).reshape(-1)
            genders = np.asarray(output_map["gender"]).reshape(-1)

        results = []

        for (x, y, w, h), age, gender_score in zip(
            faces, ages, genders
        ):
            age = max(0, float(age))

            # UTKFace labels: 0 = Male, 1 = Female.
            gender = "Female" if float(gender_score) >= 0.5 else "Male"

            results.append({
                "box": (int(x), int(y), int(w), int(h)),
                "age": round(age),
                "gender": gender,
                "senior": age > SENIOR_PREDICTION_THRESHOLD
            })

        return results
