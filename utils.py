from typing import List, Dict
from minsearch import Index
from langchain_ollama import OllamaLLM

# Initialisation du modèle LLM
llm = OllamaLLM(model="llama3:latest")


def search(query: str, index: Index, num_results: int = 3) -> List[Dict]:
    """
    Recherche des documents pertinents dans l'index Minsearch.

    Args:
        query (str): La requête utilisateur.
        index (Index): L'index Minsearch contenant les recettes.
        num_results (int): Nombre de résultats à retourner (par défaut 3).

    Returns:
        List[Dict]: Liste de documents correspondant à la requête.
    """
    results = index.search(
        query=query,
        boost_dict={},  # possibilité de boost selon des champs
        num_results=num_results
    )
    return results


def build_prompt(query: str, results: List[Dict]) -> str:
    """
    Construit un prompt professionnel pour le LLM en utilisant les résultats de recherche.

    Args:
        query (str): La question posée par l'utilisateur.
        results (List[Dict]): Documents pertinents issus de l'index.

    Returns:
        str: Prompt formaté prêt à être envoyé au LLM.
    """
    prompt_template = """
Contexte :
Tu es un assistant culinaire expert, spécialisé dans la cuisine africaine et internationale. 
Tu disposes d'un vaste dataset de recettes comprenant les informations suivantes pour chaque plat : nom, ingrédients, temps de cuisson, niveau de difficulté, type de repas, régime alimentaire, pays ou région d'origine, temps total de préparation, et épices ou saveurs principales.

Instructions : 
- Utilise uniquement les informations fournies dans le CONTEXTE pour répondre à la QUESTION.
- Fournis une réponse claire, structurée et pratique.
- Ajoute des conseils ou un guide étape par étape pour aider l'utilisateur à accomplir sa tâche culinaire.

QUESTION : 
{question}

CONTEXTE : 
{context}
""".strip()

    entry_template = """
Nom de la recette : {Recette}
Ingrédients : {Ingrédients}
Temps de cuisson : {Temps_de_cuisson}
Niveau de difficulté : {Niveau_de_difficulté}
Type de repas : {Type_de_repas}
Régime alimentaire : {Régime}
Pays ou région d'origine : {Pays}
Temps total de préparation : {Temps_total}
Épices et saveurs principales : {Épices/Saveurs}
""".strip()

    context = "\n\n".join([entry_template.format(**doc) for doc in results])
    prompt = prompt_template.format(question=query, context=context)
    return prompt


def rag(query: str, index: Index, num_results: int = 3) -> str:
    """
    Répond à la question utilisateur en combinant recherche et LLM (RAG).

    Args:
        query (str): La question de l'utilisateur.
        index (Index): L'index contenant les recettes.
        num_results (int): Nombre de documents à récupérer pour le contexte.

    Returns:
        str: Réponse générée par le LLM.
    """
    results = search(query, index, num_results)
    if not results:
        return "Aucune recette pertinente n'a été trouvée pour votre requête."

    prompt = build_prompt(query, results)
    print("✅ Traitement en cours...")
    answer = llm.invoke(prompt)
    return answer
