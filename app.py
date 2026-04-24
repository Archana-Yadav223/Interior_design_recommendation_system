import os
import numpy as np
import pickle
from flask import Flask, render_template, request
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity
import hashlib

app = Flask(__name__)

# Load model
model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')

# Load features
features_list = pickle.load(open("features.pkl", "rb"))
image_paths = pickle.load(open("image_paths.pkl", "rb"))

UPLOAD_FOLDER = "static/uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ---------------- FEATURE EXTRACTION ----------------
def extract_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array).flatten()
    return features

# ---------------- FIND SIMILAR IMAGES ----------------
def find_similar_images(query_img_path, category, top_n=5):
    query_features = extract_features(query_img_path)

    similarities = cosine_similarity([query_features], features_list)[0]
    indices = similarities.argsort()[::-1]

    results = []
    seen_similarities = []

    for i in indices:
        path = image_paths[i]
        score = similarities[i]

        # ✅ FILTER BY CATEGORY
        if category not in path:
            continue

        if any(abs(score - s) < 0.0001 for s in seen_similarities):
            continue

        seen_similarities.append(score)

        clean_path = path.replace("\\", "/")
        relative_path = clean_path.replace("static/", "")

        score_percent = round(score * 100, 2)
        results.append((relative_path, score_percent))

        if len(results) >= top_n:
            break

    return results
# ---------------- OBJECT DETECTION (simple) ----------------
def detect_objects(img_path):
    objects = []
    name = img_path.lower()

    if "sofa" in name or "living" in name:
        objects.append("sofa")

    if "bed" in name or "bedroom" in name:
        objects.append("bed")

    if "bathroom" in name:
        objects.append("table")

    if "kitchen" in name:
        objects.append("table")

    return objects


def get_image_hash(path):
    with open(path,"rb") as f:
        return 
    hashlib.md5(f.read()).hexdigest()

# ---------------- SMART LINKS ----------------
def get_smart_links(category):
    links = []

    if category == "living_room":
        links.append(("Buy Sofa", "https://www.amazon.in/s?k=sofa"))
        links.append(("Buy Coffee Table", "https://www.amazon.in/s?k=coffee+table"))

    elif category == "bedroom":
        links.append(("Buy Bed", "https://www.amazon.in/s?k=bed"))
        links.append(("Buy Wardrobe", "https://www.amazon.in/s?k=wardrobe"))

    elif category == "bathroom":
        links.append(("Buy Wash Basin", "https://www.amazon.in/s?k=wash+basin"))
        links.append(("Buy Mirror", "https://www.amazon.in/s?k=bathroom+mirror"))

    elif category == "kitchen":
        links.append(("Buy Kitchen Cabinet", "https://www.amazon.in/s?k=kitchen+cabinet"))
        links.append(("Buy Dining Table", "https://www.amazon.in/s?k=dining+table"))

    return links

# ---------------- ROUTE ----------------
@app.route("/", methods=["GET", "POST"])
def index():
    query_path = None
    results = []
    suggestions = []
    category = None

    if request.method == "POST":
        file = request.files["image"]
        category = request.form.get("category")

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        query_path = "uploads/" + file.filename

        # ✅ use filepath ONLY here
        results = find_similar_images(filepath, category)

        suggestions = get_smart_links(category)

    return render_template(
        "index.html",
        query=query_path,
        results=results,
        suggestions=suggestions,
        category=category
    )

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)
