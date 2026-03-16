import streamlit as st
from ultralytics import YOLO
from PIL import Image
import uuid
import json
import os

# -------------------------
# STREAMLIT SETTINGS
# -------------------------

st.set_page_config(page_title="Fundkiste", layout="wide")

st.title("🏫 KI-Fundkiste der Schule")

# -------------------------
# DATENORDNER ERSTELLEN
# -------------------------

UPLOAD_FOLDER = "uploads"
DATA_FILE = "data.json"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

if not os.path.exists(DATA_FILE):
    with open(DATA_FILE, "w") as f:
        json.dump([], f)

# -------------------------
# YOLO MODELL LADEN
# -------------------------

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# -------------------------
# BILD ERKENNEN
# -------------------------

def detect_object(image):

    results = model(image)

    if len(results[0].boxes) > 0:
        cls_id = int(results[0].boxes.cls[0])
        label = model.names[cls_id]
        return label
    else:
        return "Unbekannt"

# -------------------------
# BILD UPLOAD
# -------------------------

uploaded_file = st.file_uploader(
    "Foto des Fundstücks hochladen",
    type=["jpg", "jpeg", "png"]
)

beschreibung = st.text_input("Beschreibung")
fundort = st.text_input("Fundort")

# -------------------------
# FUNDSTÜCK SPEICHERN
# -------------------------

if st.button("Fundstück speichern"):

    if uploaded_file is None:
        st.error("Bitte ein Bild hochladen")

    else:

        image = Image.open(uploaded_file)

        # YOLO Analyse
        kategorie = detect_object(image)

        filename = f"{uuid.uuid4()}.jpg"
        filepath = os.path.join(UPLOAD_FOLDER, filename)

        image.save(filepath)

        # Daten laden
        with open(DATA_FILE, "r") as f:
            data = json.load(f)

        # Neues Fundstück
        new_item = {
            "bild": filepath,
            "kategorie": kategorie,
            "beschreibung": beschreibung,
            "fundort": fundort,
            "status": "Offen"
        }

        data.append(new_item)

        # Speichern
        with open(DATA_FILE, "w") as f:
            json.dump(data, f)

        st.success(f"Fundstück gespeichert! Kategorie: {kategorie}")

# -------------------------
# FUNDSTÜCKE ANZEIGEN
# -------------------------

st.header("Gefundene Gegenstände")

with open(DATA_FILE, "r") as f:
    items = json.load(f)

for item in reversed(items):

    st.image(item["bild"], width=200)

    st.write("Kategorie:", item["kategorie"])
    st.write("Beschreibung:", item["beschreibung"])
    st.write("Fundort:", item["fundort"])
    st.write("Status:", item["status"])

    st.markdown("---")
