from pathlib import Path

import joblib
import streamlit as st

MODEL_PATH = Path("models/tfidf_logreg.joblib")

st.set_page_config(page_title="Multilingual Fake News Detection", page_icon="🔎")
st.title("Multilingual Fake News Detection")
st.caption("Prototype de recherche — classification du texte, pas vérification factuelle.")

if not MODEL_PATH.exists():
    st.warning("Aucun modèle entraîné. Ajoutez le dataset puis lancez le script d'entraînement décrit dans README.md.")
    st.stop()

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()
language = st.selectbox("Langue du texte", ["English", "Français", "العربية", "Autre / inconnue"])
text = st.text_area("Collez le texte de l'article", height=240, placeholder="Saisissez le texte à analyser…")

if st.button("Analyser le texte", type="primary"):
    if len(text.strip()) < 30:
        st.error("Veuillez saisir un texte plus long (au moins 30 caractères).")
    else:
        prediction = model.predict([text])[0]
        st.subheader("Résultat du modèle")
        if prediction == "fake":
            st.warning("Classe prédite : FAKE")
        else:
            st.success("Classe prédite : REAL")

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba([text])[0]
            classes = model.classes_
            st.write("Scores estimés par le classifieur (ce ne sont pas des probabilités de vérité).")
            st.bar_chart({str(label): float(prob) for label, prob in zip(classes, probabilities)})

        st.info(
            "La langue choisie est informative pour l'instant : ce modèle TF-IDF n'est pas automatiquement "
            "multilingue. Il doit être entraîné sur des exemples représentatifs de cette langue. "
            "N'utilisez pas cette sortie comme preuve qu'une information est vraie ou fausse."
        )
