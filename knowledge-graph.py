import networkx as nx
import os
import json
from datetime import datetime
from transformers import pipeline
from github import Github
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import matplotlib.pyplot as plt

TOKEN = os.getenv("TOKEN")
g = Github(TOKEN)
nlp = pipeline("feature-extraction", model="distilbert-base-cased")
semantic_model = pipeline("feature-extraction", model="distilbert-base-cased")
code_description_model = pipeline("text2text-generation", model="t5-base")

def get_org_repos(org_name):
    try:
        org = g.get_organization(org_name)
        return [repo.full_name for repo in org.get_repos() if repo.size > 0]  # Ignora repositórios vazios
    except Exception as e:
        print(f"Erro ao buscar repositórios para a organização {org_name}: {e}")
        return []
    
def generate_description(code):
    input_text = f"describe the code: {code}"
    output = code_description_model(input_text, max_length=150, num_return_sequences=1)
    return output[0]['generated_text']

def enrich_repository_data(repo_data_list):
    for repo in repo_data_list:
        embeddings = semantic_model(repo["description"])
        repo["semantic_embedding"] = embeddings[0][0] if embeddings else []
    return repo_data_list

def fetch_repo_data(repo_name):
    try:
        repo = g.get_repo(repo_name)
        code_content = ""
        for file in repo.get_contents("")[:5]:  # Limitar a 5 arquivos
            if file.type == "file":
                code_content += repo.get_contents(file.path).decoded_content.decode() + "\n"

        data = {
            "name": repo.name,
            "description": repo.description or "",
            "code_description": generate_description(code_content),  # Gera a descrição do código
            "topics": repo.get_topics(),
            "language": repo.language,
            "stars": repo.stargazers_count,
            "contributors": [contributor.login for contributor in repo.get_contributors()],
            "forks": repo.forks_count,
            "updated_at": repo.updated_at.isoformat() if repo.updated_at else None,
            "size": repo.size,
        }
        return data
    except Exception as e:
        print(f"Erro ao buscar dados do repositório {repo_name}: {e}")
        return None

@lru_cache(maxsize=128)
def get_embeddings(text):
    if text:
        return nlp(text)[0][0]
    return []

def calculate_similarity(embedding1, embedding2, stars1, stars2, language1, language2, topics1, topics2, 
                         contributors1, contributors2, forks1, forks2, updated_at1, updated_at2, size1, size2):
    # Simplificação do cálculo de similaridade
    description_similarity = sum(e1 * e2 for e1, e2 in zip(embedding1, embedding2))
    star_similarity = min(stars1, stars2) / max(stars1, stars2) if max(stars1, stars2) > 0 else 0
    language_similarity = 1 if language1 == language2 else 0
    topic_similarity = len(set(topics1).intersection(set(topics2))) / max(len(topics1), len(topics2), 1)
    contributor_similarity = len(set(contributors1).intersection(set(contributors2))) / max(len(contributors1), len(contributors2), 1)
    fork_similarity = min(forks1, forks2) / max(forks1, forks2) if max(forks1, forks2) > 0 else 0
    size_similarity = min(size1, size2) / max(size1, size2) if max(size1, size2) > 0 else 0

    overall_similarity = (
        0.3 * description_similarity +
        0.15 * star_similarity +
        0.1 * language_similarity +
        0.1 * topic_similarity +
        0.1 * contributor_similarity +
        0.05 * fork_similarity +
        0.1 * size_similarity
    )
    return overall_similarity

def process_repositories(repos):
    kg = nx.Graph()
    repo_data_list = []

    with ThreadPoolExecutor(max_workers=5) as executor:  # Limita a 5 threads
        repo_data_list = list(executor.map(fetch_repo_data, repos))

    repo_data_list = enrich_repository_data(repo_data_list)

    embeddings_map = {repo_data["name"]: repo_data["semantic_embedding"] for repo_data in repo_data_list if repo_data}

    for i, repo_data in enumerate(repo_data_list):
        if repo_data:
            kg.add_node(repo_data["name"], **repo_data)
            for contributor in repo_data["contributors"]:
                kg.add_node(contributor, type="contributor")
                kg.add_edge(repo_data["name"], contributor, weight=1)

    for i, repo_data in enumerate(repo_data_list):
        if repo_data:
            for other_data in repo_data_list[i + 1:]:  # Simplifica iteração
                if other_data:
                    similarity = calculate_similarity(
                        embeddings_map[repo_data["name"]],
                        embeddings_map[other_data["name"]],
                        repo_data["stars"], other_data["stars"],
                        repo_data["language"], other_data["language"],
                        repo_data["topics"], other_data["topics"],
                        repo_data["contributors"], other_data["contributors"],
                        repo_data["forks"], other_data["forks"],
                        repo_data["updated_at"], other_data["updated_at"],
                        repo_data["size"], other_data["size"]
                    )
                    if similarity > 0.5:
                        kg.add_edge(repo_data["name"], other_data["name"], weight=similarity)
    return kg, repo_data_list

def save_to_json(data, filename="repo_data.json"):
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Dados salvos em {filename}")

def visualize_knowledge_graph(kg):
    plt.figure(figsize=(14, 10))
    
    pos = nx.spring_layout(kg, k=0.5)
    node_sizes = [kg.nodes[node]["stars"] * 10 if "stars" in kg.nodes[node] else 50 for node in kg.nodes]
    node_colors = ['lightblue' if kg.nodes[node].get("language") is None else 'orange' for node in kg.nodes]

    nx.draw(kg, pos, with_labels=True, node_size=node_sizes, node_color=node_colors, font_size=10, font_weight="bold", edge_color="#BBBBBB", alpha=0.7)
    plt.title("Knowledge Graph")
    plt.show()

def main():
    org_name = input("Digite o nome da organização: ")
    repos = get_org_repos(org_name)
    
    if repos:
        knowledge_graph, repo_data_list = process_repositories(repos)
        save_to_json(repo_data_list)
        visualize_knowledge_graph(knowledge_graph)
    else:
        print(f"Nenhuma organização encontrada com o nome: {org_name}.")

if __name__ == "__main__":
    main()
