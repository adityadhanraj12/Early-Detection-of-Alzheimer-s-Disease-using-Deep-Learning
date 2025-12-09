import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# ----------------------------------------------------
# 🎨 PAGE CONFIGURATION
# ----------------------------------------------------
st.set_page_config(
    page_title="🧠 Alzheimer’s Stage Predictor",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
    <style>
        .main {background-color: #f7f6fb;}
        div.stButton > button:first-child {
            background-color: #6366f1;
            color: white;
            border-radius: 8px;
        }
        .banner {
            background: linear-gradient(90deg, #6366f1, #60a5fa);
            color: white;
            padding: 25px;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 20px;
        }
        .stFileUploader label {color: #374151; font-weight: bold;}
    </style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# 🌐 SIDEBAR NAVIGATION
# ----------------------------------------------------
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Go to:", ["Home", "MRI Prediction", "About"])

# ----------------------------------------------------
# 🏠 HOME PAGE
# ----------------------------------------------------
if page == "Home":
    st.markdown("""
    <div class='banner'>
        <h2>🧠 Alzheimer’s Stage Predictor</h2>
        <p>Upload a brain MRI to predict dementia stage and explore in-depth information.</p>
    </div>
    """, unsafe_allow_html=True)

    st.write("### What is Alzheimer’s Disease?")
    st.markdown("""
    Alzheimer’s is a **neurodegenerative disorder** that affects memory, thinking, and reasoning.  
    It gradually damages brain cells and neural pathways, leading to dementia and cognitive decline.  
    It’s the most common form of dementia, accounting for **60–80% of cases** globally.
    """)

    st.image("https://upload.wikimedia.org/wikipedia/commons/7/7f/Alzheimer_disease_brain_comparison.jpg",
             caption="Brain comparison: Healthy vs. Alzheimer’s brain", use_container_width=True)

    st.divider()
    st.subheader("🧠 Stages of Alzheimer’s Disease")
    st.markdown("""
    1. **Non-Demented:** Normal cognition, no memory loss.  
    2. **Very Mild Dementia:** Subtle memory issues but independence maintained.  
    3. **Mild Dementia:** Daily tasks affected; noticeable cognitive decline.  
    4. **Moderate Dementia:** Requires care and supervision.
    """)

    st.success("💡 Early diagnosis allows effective management and better quality of life.")


# ----------------------------------------------------
# 🔍 MRI PREDICTION PAGE
# ----------------------------------------------------
elif page == "MRI Prediction":
    st.markdown("""
    <div class='banner'>
        <h2>🔬 Alzheimer’s Stage Prediction</h2>
        <p>Upload a brain MRI scan to predict the dementia stage and get detailed care insights.</p>
    </div>
    """, unsafe_allow_html=True)

    # Load model
    @st.cache_resource
    def load_model():
        return tf.keras.models.load_model("alzheimers_cnn_deep_model.h5")

    model = load_model()

    labels = {
        0: "Mild Dementia",
        1: "Moderate Dementia",
        2: "Non-Demented",
        3: "Very Mild Dementia"
    }

    # Detailed stage info
    stage_info = {
        "Non-Demented": {
            "stage": "Non-Demented",
            "description": "No signs of cognitive impairment or memory loss. Individual functions independently.",
            "symptoms": ["Normal functioning; no visible memory issues."],
            "progression": "Cognitive abilities remain stable; regular monitoring is advised.",
            "risk_factors": ["Age 65+", "Family history", "Sedentary lifestyle", "High BP or cholesterol"],
            "care_tips": [
                "Stay mentally active (puzzles, reading).",
                "Exercise regularly and eat a balanced diet.",
                "Keep social connections and avoid isolation."
            ],
            "legal_considerations": "Early estate and medical planning recommended.",
            "support_resources": [
                {"label": "Alzheimer's Association", "url": "https://www.alz.org"},
                {"label": "National Institute on Aging", "url": "https://www.nia.nih.gov"}
            ],
            "reference": "https://www.alz.org/alzheimers-dementia/stages"
        },
        "Very Mild Dementia": {
            "stage": "Very Mild Dementia (Early Stage)",
            "description": "Minor memory lapses and cognitive changes; still independent.",
            "symptoms": [
                "Occasional forgetfulness",
                "Slight difficulty recalling names or words",
                "Mild confusion in unfamiliar settings"
            ],
            "progression": "May remain stable for years with lifestyle changes.",
            "risk_factors": ["Mild Cognitive Impairment", "Head injury", "Chronic stress"],
            "care_tips": [
                "Use reminders and notes to stay organized.",
                "Regular mental exercises to boost cognition.",
                "Track symptoms and consult healthcare providers early."
            ],
            "legal_considerations": "Start preparing medical and financial directives.",
            "support_resources": [
                {"label": "Early Stage Caregiving", "url": "https://www.alz.org/help-support/caregiving/stages-behaviors/early-stage"}
            ],
            "reference": "https://www.mayoclinic.org/diseases-conditions/alzheimers-disease/in-depth/alzheimers-stages/art-20048448"
        },
        "Mild Dementia": {
            "stage": "Mild Dementia (Middle Stage Early)",
            "description": "Memory and reasoning difficulties affect daily life. Friends and family notice changes.",
            "symptoms": [
                "Short-term memory loss",
                "Trouble managing finances or planning",
                "Mood and behavior changes"
            ],
            "progression": "Worsens over 2–4 years; supervision becomes necessary.",
            "risk_factors": ["Progressive brain changes", "Uncontrolled diabetes", "Social isolation"],
            "care_tips": [
                "Create routines and keep a familiar environment.",
                "Supervise complex activities like cooking or finances.",
                "Ensure emotional and social support."
            ],
            "legal_considerations": "Finalize power of attorney and financial documents.",
            "support_resources": [
                {"label": "Alzheimer’s Caregiver Center", "url": "https://www.alz.org/help-support/caregiving"},
                {"label": "Caregiver Resources", "url": "https://www.alz.org/help-support/resources"}
            ],
            "reference": "https://www.alz.org/alzheimers-dementia/stages"
        },
        "Moderate Dementia": {
            "stage": "Moderate Dementia (Advanced Middle Stage)",
            "description": "Severe cognitive decline; daily assistance required.",
            "symptoms": [
                "Difficulty recognizing family/friends",
                "Disorientation and confusion",
                "Behavioral changes like agitation or suspicion"
            ],
            "progression": "Lasts 2–10 years; full-time care often needed.",
            "risk_factors": ["Infections", "Medication side effects", "Wandering"],
            "care_tips": [
                "24/7 supervision for safety.",
                "Use GPS trackers and home safety measures.",
                "Monitor for infections or pain."
            ],
            "legal_considerations": "Activate medical and legal directives.",
            "support_resources": [
                {"label": "NIH Fact Sheet", "url": "https://www.nia.nih.gov/health/alzheimers-disease-fact-sheet"},
                {"label": "Alzheimer’s Safety Center", "url": "https://www.alz.org/help-support/caregiving/daily-care/safety"}
            ],
            "reference": "https://www.alz.org/help-support/caregiving/stages-behaviors/middle-stage"
        }
    }

    # Upload MRI
    uploaded_file = st.file_uploader("📤 Upload a Brain MRI Image", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="🧩 Uploaded MRI Image", use_container_width=True)

        img = image.resize((128,128))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        pred = model.predict(img_array)
        predicted_class = np.argmax(pred, axis=1)[0]
        predicted_label = labels[predicted_class]

        st.success(f"### 🧠 Prediction: {predicted_label}")

        # Detailed Info
        info = stage_info.get(predicted_label, {})
        st.header(f"Stage: {info.get('stage','-')}")
        st.markdown(f"**Description:** {info.get('description','-')}")

        if info.get("symptoms"):
            st.subheader("⚠️ Common Symptoms")
            for s in info["symptoms"]:
                st.write(f"- {s}")

        if info.get("progression"):
            st.subheader("⏳ Progression & Timeline")
            st.write(info["progression"])

        if info.get("risk_factors"):
            st.subheader("🧬 Risk Factors")
            for r in info["risk_factors"]:
                st.write(f"- {r}")

        if info.get("care_tips"):
            st.subheader("❤️ Care and Support Recommendations")
            for c in info["care_tips"]:
                st.write(f"- {c}")

        if info.get("legal_considerations"):
            st.subheader("⚖️ Legal & Financial Considerations")
            st.write(info["legal_considerations"])

        if info.get("support_resources"):
            st.subheader("🌐 Support Resources & Helplines")
            for res in info["support_resources"]:
                st.write(f"[{res['label']}]({res['url']})")

        if info.get("reference"):
            st.markdown(f"**Reference:** [Read more here]({info['reference']})")

        st.markdown("---")
        st.info("📞 24/7 Alzheimer’s Association Helpline: 1-800-272-3900")
    else:
        st.warning("Please upload an MRI image to get a prediction.")

# ----------------------------------------------------
# ℹ️ ABOUT PAGE
# ----------------------------------------------------
elif page == "About":
    st.markdown("""
    <div class='banner'>
        <h2>📘 About This Project</h2>
        <p>Developed using Deep Learning and Streamlit.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    **Developers:**  
    - 🧑‍💻 *Pisini Joel*  
    - 🧑‍💻 *Aditya Dhanraj*  

    **Purpose:**  
    This project leverages a **Convolutional Neural Network (CNN)** trained on MRI images  
    to predict the **stage of Alzheimer’s disease** for early detection and care planning.

    **Frameworks Used:**  
    - TensorFlow / Keras  
    - NumPy, PIL  
    - Streamlit for interactive UI  

    **Disclaimer:**  
    This app is for **educational and awareness purposes only**  
    and not a substitute for professional medical advice.
    """)

    st.divider()
    st.caption("© 2025 | Alzheimer’s Stage Predictor | Deep Learning Project")
