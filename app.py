
import streamlit as st
import tensorflow as tf
import numpy as np
import json
import urllib.request
from pathlib import Path
from PIL import Image

# Project paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "plant_disease_model.keras"
LABEL_PATH = BASE_DIR / "documentation" / "labels.json"

MODEL_URL = (
    "https://huggingface.co/ashikaasriarun/"
    "MobileNet-V2-PlantVillage/resolve/main/"
    "best_mnv2_pv_original.keras"
)

st.set_page_config(
    page_title="Crop Disease AI",
    page_icon="🌱",
    layout="wide"
)

# Download model if it does not exist
@st.cache_resource
def load_model():

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH.exists():
        st.info("Downloading AI model for first-time use...")

        try:
            urllib.request.urlretrieve(
                MODEL_URL,
                str(MODEL_PATH)
            )
        except Exception as e:
            if MODEL_PATH.exists():
                MODEL_PATH.unlink()
            raise RuntimeError(
                f"Model download failed: {e}"
            )

    return tf.keras.models.load_model(
        str(MODEL_PATH),
        compile=False
    )


# Load labels
with open(LABEL_PATH, "r", encoding="utf-8") as f:
    CLASS_NAMES = json.load(f)


# Advisory database
ADVISORY = {
    "powdery mildew": (
        "Avoid overcrowding plants. Improve air circulation and remove "
        "severely affected leaves where appropriate. Ask an agricultural "
        "officer about locally approved treatment.",
        "செடிகள் நெருக்கமாக வளர்வதைத் தவிர்க்கவும். காற்றோட்டத்தை "
        "மேம்படுத்தவும். பாதிக்கப்பட்ட இலைகளை கவனமாக அகற்றவும். "
        "உள்ளூர் வேளாண் அலுவலரிடம் உரிய சிகிச்சையை கேளுங்கள்."
    ),
    "early blight": (
        "Avoid wetting foliage during irrigation. Remove affected plant "
        "debris and monitor nearby leaves.",
        "இலைகள் நனையும்படி நீர் பாய்ச்சுவதைத் தவிர்க்கவும். "
        "பாதிக்கப்பட்ட தாவரக் கழிவுகளை அகற்றி மற்ற இலைகளையும் கண்காணிக்கவும்."
    ),
    "late blight": (
        "Inspect plants frequently. Avoid handling wet plants and seek "
        "prompt advice from an agricultural expert.",
        "செடிகளை அடிக்கடி கண்காணிக்கவும். ஈரமான செடிகளை கையாள்வதைத் "
        "தவிர்த்து உடனடியாக வேளாண் நிபுணரிடம் ஆலோசனை பெறவும்."
    ),
    "rust": (
        "Remove badly affected leaves where appropriate and avoid "
        "overhead watering. Seek local expert advice.",
        "அதிகம் பாதிக்கப்பட்ட இலைகளை தேவையானபோது அகற்றவும். "
        "மேலிருந்து நீர் தெளிப்பதைத் தவிர்த்து நிபுணர் ஆலோசனை பெறவும்."
    ),
    "bacterial spot": (
        "Avoid working among wet plants. Use clean tools and seek "
        "agricultural advice before treatment.",
        "ஈரமான செடிகளுக்கு இடையில் வேலை செய்வதைத் தவிர்க்கவும். "
        "சுத்தமான கருவிகளைப் பயன்படுத்தி சிகிச்சைக்கு முன் வேளாண் "
        "ஆலோசனை பெறவும்."
    ),
    "healthy": (
        "No disease symptom was identified by this prediction. "
        "Continue regular monitoring and good field hygiene.",
        "இந்த கணிப்பில் நோய் அறிகுறி கண்டறியப்படவில்லை. "
        "தொடர்ந்து செடிகளை கண்காணித்து வயலை சுத்தமாக வைத்திருக்கவும்."
    )
}


# Language selection
language = st.selectbox(
    "Select Language / மொழியை தேர்வு செய்க",
    ["English", "தமிழ்"]
)

st.title("🌱 Offline Crop Disease Detection & Advisory")
st.subheader("Bharat AI-SoC Student Challenge 2026–27")
st.write("Problem Statement 1 – Offline crop disease detection")


# Load AI model
try:
    with st.spinner("Loading AI model..."):
        model = load_model()

    st.success("AI Model Loaded Successfully")

except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()


st.divider()
st.header("📷 Upload Crop Leaf Image")

uploaded_file = st.file_uploader(
    "Choose a leaf image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Leaf",
        width=350
    )

    if st.button("🔍 Detect Disease", type="primary"):

        with st.spinner("Analysing image..."):

            resized = image.resize((160, 160))

            arr = np.array(
                resized,
                dtype=np.float32
            )

            arr = np.expand_dims(arr, axis=0)

            prediction = model.predict(
                arr,
                verbose=0
            )[0]

            index = int(np.argmax(prediction))

            confidence = float(prediction[index]) * 100

            raw_label = CLASS_NAMES[index]

            crop, condition = raw_label.split("___")

            crop = crop.replace("_", " ")
            condition = condition.replace("_", " ")


        st.divider()
        st.header("Detection Result")

        st.write("**Crop / பயிர்:**", crop)

        st.write(
            "**Predicted condition / கணிக்கப்பட்ட நிலை:**",
            condition
        )

        st.metric(
            "Model Confidence",
            f"{confidence:.2f}%"
        )


        # Prototype threshold: 40%
        if confidence < 40:

            st.warning(
                "Low confidence. This is only a tentative prediction. "
                "Please upload a clear image and seek expert confirmation."
                if language == "English"
                else "கணிப்பு நம்பகத்தன்மை குறைவாக உள்ளது. "
                "தெளிவான படத்தை பதிவேற்றி வேளாண் நிபுணரிடம் உறுதிப்படுத்தவும்."
            )

            st.subheader(
                "General Precautions / பொதுவான முன்னெச்சரிக்கைகள்"
            )

            if language == "English":

                st.write(
                    "- Take a clear photo in natural light.\n"
                    "- Capture both sides of the leaf.\n"
                    "- Check other leaves and nearby plants.\n"
                    "- Do not apply chemicals based only on this result."
                )

            else:

                st.write(
                    "- இயற்கை வெளிச்சத்தில் தெளிவான படம் எடுக்கவும்.\n"
                    "- இலையின் இரு பக்கங்களையும் படம் எடுக்கவும்.\n"
                    "- அருகிலுள்ள செடிகளையும் பரிசோதிக்கவும்.\n"
                    "- இந்த கணிப்பை மட்டும் வைத்து மருந்து தெளிக்க வேண்டாம்."
                )

        else:

            st.success(
                "Prototype prediction (40% threshold). "
                "Expert confirmation is recommended."
            )

            key = condition.lower()

            advice = ADVISORY.get(key)

            if advice:

                st.subheader(
                    "Farmer Advisory / விவசாயி ஆலோசனை"
                )

                st.write(
                    advice[0] if language == "English"
                    else advice[1]
                )

            else:

                st.info(
                    "Please consult a local agricultural officer for "
                    "verified crop-specific guidance."
                    if language == "English"
                    else "சரியான பயிர் சார்ந்த ஆலோசனைக்கு உள்ளூர் "
                    "வேளாண் அலுவலரை அணுகவும்."
                )


        st.caption(
            "AI prediction is preliminary, not a confirmed diagnosis. "
            "No pesticide recommendation is made by this prototype."
        )

else:

    st.info("Upload a crop leaf image to start.")


st.divider()

st.caption("Prototype | MobileNetV2 | PlantVillage 38 classes")

st.caption(
    "Model runs locally after the first download. "
    "Internet is required for the initial model download."
)
