import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# Developer credits at the top
st.markdown("<h3 style='text-align: center;'>Developed by <b>Pisini Joel</b> and <b>Aditya Dhanraj</b></h3>", unsafe_allow_html=True)
st.title("Alzheimer's Stage Predictor & Information Portal")
st.caption("Upload a brain MRI slice to predict dementia stage, with in-depth stage guidance.")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('alzheimers_cnn_deep_model.h5')

model = load_model()

labels = {
    0: 'mild_dementia',
    1: 'moderate_dementia',
    2: 'non_demented',
    3: 'very_mild_dementia'
}

stage_info = {
    'non_demented': {
        "stage": "Non-demented (No apparent impairment)",
        "description": "No signs of cognitive impairment or memory loss. The individual is able to function independently, with normal memory and reasoning.",
        "symptoms": [],
        "progression": "Cognitive abilities remain stable. Regular monitoring is advised since risk factors can accumulate over time.",
        "risk_factors": [
            "Advanced age (65+)",
            "Family history of dementia",
            "Sedentary lifestyle",
            "Poor cardiovascular health",
            "Diabetes",
            "High blood pressure"
        ],
        "care_tips": [
            "Continue a healthy lifestyle with regular mental, physical, and social activities.",
            "Routine medical evaluations to detect any early changes.",
            "Maintain a balanced diet (Mediterranean-style diet recommended).",
            "Regular exercise (30 minutes, 5 days a week).",
            "Stay socially engaged and mentally stimulated."
        ],
        "legal_considerations": "No immediate legal needs, but early estate planning and creation of advanced directives are always beneficial for future preparedness.",
        "support_resources": [
            {"label": "Alzheimer's Association Main Site", "url": "https://www.alz.org"},
            {"label": "National Institute on Aging - Alzheimer's Info", "url": "https://www.nia.nih.gov/health/alzheimers-and-dementia"},
            {"label": "Alzheimer's Prevention Tips", "url": "https://www.alzheimers.gov/life-with-dementia"}
        ],
        "additional_info": "Prevention strategies include cognitive training, social engagement, and managing cardiovascular risk factors.",
        "reference": "https://www.alz.org/alzheimers-dementia/stages"
    },
    'very_mild_dementia': {
        "stage": "Very Mild Dementia (Early Stage)",
        "description": "Minor memory lapses and cognitive changes that may not be immediately noticeable to others. Individuals can still function independently in most daily activities.",
        "symptoms": [
            "Occasional memory lapses",
            "Mild word-finding difficulty",
            "Slight trouble with complex tasks",
            "Forgetting recent conversations or events",
            "Difficulty remembering names of new people",
            "Trouble with planning and organization"
        ],
        "progression": "Progression may be slow or remain stable for years. Early intervention and lifestyle modifications can help slow progression. Regular cognitive assessments recommended every 6-12 months.",
        "risk_factors": [
            "Mild cognitive impairment (MCI) history",
            "Previous brain injury or trauma",
            "Genetic predisposition (APOE gene variants)",
            "Chronic stress and depression",
            "Sleep disorders"
        ],
        "care_tips": [
            "Encourage regular mental stimulation like puzzles, reading, or learning new skills.",
            "Maintain consistent daily routines and adequate sleep (7-9 hours).",
            "Monitor for changes and keep a symptoms diary.",
            "Use memory aids like calendars, notes, and smartphone reminders.",
            "Stay physically active with regular exercise.",
            "Maintain social connections and activities."
        ],
        "legal_considerations": "Ideal time to start legal and financial planning, including power of attorney, advance care directives, and healthcare proxies while the individual can still participate fully in decisions.",
        "support_resources": [
            {"label": "Alzheimer's Association Stages Guide", "url": "https://www.alz.org/alzheimers-dementia/stages"},
            {"label": "Early Stage Caregiving Resources", "url": "https://www.alz.org/help-support/caregiving/stages-behaviors/early-stage"},
            {"label": "National Institute on Aging Resources", "url": "https://www.nia.nih.gov/health/alzheimers-disease-fact-sheet"},
            {"label": "24/7 Alzheimer's Helpline", "url": "https://www.alz.org/help-support/resources"}
        ],
        "additional_info": "Early diagnosis allows for better planning and access to treatments that may help slow progression. Consider joining clinical trials for potential new treatments.",
        "reference": "https://www.mayoclinic.org/diseases-conditions/alzheimers-disease/in-depth/alzheimers-stages/art-20048448"
    },
    'mild_dementia': {
        "stage": "Mild Dementia (Middle Stage - Early)",
        "description": "Noticeable memory loss and cognitive changes that interfere with daily life. Family and friends will likely notice changes, and professional intervention becomes important.",
        "symptoms": [
            "Significant short-term memory loss",
            "Frequently misplacing items",
            "Trouble with planning, organizing, and decision-making",
            "Getting lost in familiar places",
            "Difficulty managing finances and medications",
            "Problems with language and communication",
            "Changes in mood and personality",
            "Withdrawal from social activities"
        ],
        "progression": "Symptoms become more apparent and may increase over 2-4 years. Most individuals can still live independently with some support and supervision. This stage often lasts the longest.",
        "risk_factors": [
            "Progressive neurodegeneration",
            "Uncontrolled diabetes",
            "Untreated hypertension",
            "Social isolation",
            "Lack of mental stimulation"
        ],
        "care_tips": [
            "Establish and maintain consistent daily routines.",
            "Use calendars, notes, pill organizers, and digital reminders.",
            "Provide gentle supervision for complex tasks like driving, cooking, and managing finances.",
            "Encourage independence in familiar activities while ensuring safety.",
            "Create a safe, clutter-free home environment.",
            "Consider adult day programs for socialization and activities.",
            "Monitor nutrition and ensure regular meals."
        ],
        "legal_considerations": "Critical time to complete all legal and financial planning. Consider discussing driving safety, financial management, and future care preferences. May need to involve financial advisor or elder law attorney.",
        "support_resources": [
            {"label": "Alzheimer's Association Caregiving Center", "url": "https://www.alz.org/help-support/caregiving"},
            {"label": "Mayo Clinic Alzheimer's Stages", "url": "https://www.mayoclinic.org/diseases-conditions/alzheimers-disease/in-depth/alzheimers-stages/art-20048448"},
            {"label": "Caregiver Support Groups", "url": "https://www.alz.org/help-support/resources"},
            {"label": "Daily Care Planning Tools", "url": "https://www.alzheimers.gov/life-with-dementia/resources-caregivers"}
        ],
        "additional_info": "This is often when families first seek medical help. Treatment with medications like cholinesterase inhibitors may help with symptoms. Consider joining support groups for both patient and family.",
        "reference": "https://www.alz.org/alzheimers-dementia/stages"
    },
    'moderate_dementia': {
        "stage": "Moderate Dementia (Middle Stage - Advanced)",
        "description": "Significant cognitive decline with increased confusion and behavioral changes. Requires substantial daily assistance and supervision for safety and basic care.",
        "symptoms": [
            "Severe memory loss and confusion",
            "Difficulty recognizing family members and friends",
            "Problems with language and communication",
            "Wandering and getting lost",
            "Poor judgment and decision-making",
            "Sleep disturbances and day/night confusion",
            "Behavioral changes including agitation, suspicion, or hallucinations",
            "Difficulty with personal care (bathing, dressing, toileting)",
            "Problems with eating and swallowing"
        ],
        "progression": "Symptoms become severe and can change rapidly. This stage typically lasts 2-10 years. Round-the-clock supervision becomes necessary for safety and care.",
        "risk_factors": [
            "Natural disease progression",
            "Infections (UTIs, pneumonia)",
            "Medication side effects",
            "Environmental changes or stress",
            "Lack of structured care"
        ],
        "care_tips": [
            "Ensure 24/7 supervision and safety measures at home.",
            "Supervise all medications, meals, and personal hygiene.",
            "Maintain consistent daily routines and familiar environments.",
            "Use simple, clear communication and remain patient.",
            "Manage wandering with alarms, locks, or GPS devices.",
            "Address sleep issues with proper sleep hygiene.",
            "Consider professional in-home care or adult day programs.",
            "Monitor for signs of pain, infection, or other medical issues."
        ],
        "legal_considerations": "Caregivers typically need to act as legal proxies for healthcare and financial decisions. Ensure all legal documents are in place and activated as needed.",
        "support_resources": [
            {"label": "National Institute on Aging Fact Sheet", "url": "https://www.nia.nih.gov/health/alzheimers-disease-fact-sheet"},
            {"label": "Alzheimer's Association Safety Center", "url": "https://www.alz.org/help-support/caregiving/daily-care/safety"},
            {"label": "Caregiver Resources and Support", "url": "https://www.alzheimers.gov/life-with-dementia/resources-caregivers"},
            {"label": "Middle Stage Caregiving Guide", "url": "https://www.alz.org/help-support/caregiving/stages-behaviors/middle-stage"}
        ],
        "additional_info": "Consider residential care options if home care becomes unsafe or too challenging. Focus on comfort, dignity, and quality of life. Behavioral interventions may be more effective than medications.",
        "reference": "https://www.alz.org/help-support/caregiving/stages-behaviors/middle-stage"
    }
}

img_height, img_width = 128, 128

uploaded_file = st.file_uploader("Choose an MRI image (.jpg, .png)...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded MRI Slice', use_column_width=True)
    img = image.resize((img_height, img_width))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    preds = model.predict(img_array)
    pred_index = int(np.argmax(preds))
    pred_class = labels[pred_index]
    confidence = 100 * np.max(preds)
    info = stage_info[pred_class]

    st.header(f"🧠 {info['stage']}")
    st.write(f"**Model confidence:** {confidence:.2f}%")
    st.write(f"**Description:** {info['description']}")
    
    if "progression" in info:
        st.write(f"**Progression and Timeline:** {info['progression']}")
    
    if info.get('symptoms'):
        st.write("**Common symptoms:**")
        for s in info['symptoms']:
            st.write(f"• {s}")
    
    if info.get('risk_factors'):
        st.write("**Risk factors:**")
        for r in info['risk_factors']:
            st.write(f"• {r}")
    
    if info.get('care_tips'):
        st.write("**Care and support recommendations:**")
        for c in info['care_tips']:
            st.write(f"• {c}")
    
    if "legal_considerations" in info:
        st.write(f"**Legal and financial considerations:** {info['legal_considerations']}")
    
    if "additional_info" in info:
        st.write(f"**Additional information:** {info['additional_info']}")
    
    if "support_resources" in info:
        st.write("**Support resources and helplines:**")
        for res in info['support_resources']:
            st.markdown(f"• [{res['label']}]({res['url']})")
    
    st.markdown(f"**[📚 Primary reference and more information]({info['reference']})**")

    st.write("---")
    st.write("**Prediction confidence for each stage:**")
    for i, p in enumerate(preds[0]):
        st.write(f"**{labels[i].replace('_', ' ').title()}:** {p*100:.2f}%")

    # Additional resources section
    st.write("---")
    st.markdown("### 🆘 **Emergency Resources**")
    st.write("• **24/7 Alzheimer's Association Helpline:** 1-800-272-3900")
    st.write("• **Crisis support and local resources available in 200+ languages**")
    st.markdown("• **[Find local Alzheimer's Association chapter](https://www.alz.org/help-support/resources)**")

    st.markdown("### ℹ️ **General Information**")
    st.markdown("*This tool is for educational purposes only and should not replace professional medical diagnosis. Please consult with healthcare professionals for proper evaluation and treatment.*")
