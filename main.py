import pandas as pd
import streamlit as st
from minsearch import Index
from utils import search, rag

# =============================
# Chargement des données
# =============================
df = pd.read_csv("recettes_africaines.csv")
columns = df.columns


# =============================
# Streamlit UI
# =============================
st.set_page_config(page_title="Assistant Culinaire Africain", page_icon="🍲", layout="wide")
st.title("🍲 Assistant Culinaire Africain")
st.markdown("""
Bienvenue sur votre assistant culinaire interactif !  
Posez votre question et découvrez des recettes africaines adaptées à vos ingrédients, pays ou type de repas.
""")

# Sidebar pour filtres
st.sidebar.header("Filtres avancés")
selected_country = st.sidebar.selectbox("Filtrer par pays", ["Tous"] + sorted(df['Pays'].unique()))
selected_type = st.sidebar.selectbox("Type de repas", ["Tous"] + sorted(df['Type_de_repas'].unique()))
num_results = st.sidebar.slider("Nombre de recettes à considérer pour le LLM", 1, 5, 3)

# Champ de question
query = st.text_area("Votre question :", "Je n’ai que du manioc et des feuilles de manioc, quelle recette camerounaise puis-je faire ?")

# Filtrage du dataset pour l'affichage
filtered_df = df.copy()
if selected_country != "Tous":
    filtered_df = filtered_df[filtered_df['Pays'] == selected_country]
if selected_type != "Tous":
    filtered_df = filtered_df[filtered_df['Type_de_repas'] == selected_type]


index = Index(text_fields=columns)
index.fit(filtered_df.to_dict(orient='records'))
if selected_country != "Tous":
    st.success(f"Indexation complète pour {selected_country}.")

    st.success(f"Number recettes africaines pour {selected_country} enregistré dans la base de données {len(filtered_df)} ")
else:
    st.success("Indexation complète pour tous les pays.")

if st.button("🔍 Chercher la recette"):
    with st.spinner("Recherche et génération de réponse en cours..."):
        answer = rag(query, index, num_results)

    st.subheader("💡 Réponse générée par l'assistant")
    st.write(answer)

    st.subheader("📋 Recettes correspondantes")

