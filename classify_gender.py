from tf_keras.preprocessing.image import img_to_array
from tf_keras.models import load_model
import numpy as np
import argparse
import cv2
import os
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
ap.add_argument("-m", "--model", required=True, help="path to trained model file")
args = ap.parse_args()
image = cv2.imread(args.image)

if image is None:
    print("❌ Could not read input image")
    exit()
output = np.copy(image)
image = cv2.resize(image, (96, 96))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)
try:
    model = load_model(args.model)
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()
confidence = model.predict(image)[0]
classes = ["man", "woman"]
idx = np.argmax(confidence)
label = classes[idx]
label_text = "{}: {:.2f}%".format(label, confidence[idx] * 100)

cv2.putText(output, label_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
print("🖼️ Image:", args.image)
print("📊 Classes:", classes)
print("✅ Confidence:", confidence)
print("🔍 Predicted:", label_text)
cv2.imshow("Gender classification", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("gender-classification-output.jpg", output)
print("💾 Saved result as 'gender-classification-output.jpg'")