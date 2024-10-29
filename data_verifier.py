import requests
from bs4 import BeautifulSoup
from transformers import pipeline
from collections import defaultdict

def extract_text_from_url(url):
    response = requests.get(url)
    response.raise_for_status()  
    soup = BeautifulSoup(response.text, 'html.parser')
    
    paragraphs = soup.find_all('p')
    text = ' '.join([para.get_text() for para in paragraphs])
    return text

def summarize_text(text):
    summarizer = pipeline("summarization", model="t5-small")
    summary = summarizer(text, max_length=150, min_length=30, do_sample=False)
    return summary[0]['summary_text']

def categorize_news(text):
    categories = {
        "Política": ["governo", "eleições", "política", "presidente", "congresso"],
        "Economia": ["economia", "mercado", "financeiro", "investimentos", "crise"],
        "Saúde": ["saúde", "covid", "vacina", "doença", "tratamento"],
        "Tecnologia": ["tecnologia", "inovação", "software", "hardware", "internet"],
        "Entretenimento": ["filme", "música", "celebridade", "show", "artista"],
        "Esportes": ["futebol", "basquete", "atleta", "competição", "jogo"],
        "Ciência": ["ciência", "pesquisa", "descoberta", "experimento", "teoria"],
        "Meio Ambiente": ["meio ambiente", "sustentável", "ecologia", "clima", "poluição"],
        "Educação": ["educação", "escola", "universidade", "ensino", "professor"],
        "Cultura": ["cultura", "tradicional", "história", "arte", "literatura"],
        "Internacional": ["internacional", "mundo", "país", "guerra", "conflito"],
        "Negócios": ["negócios", "empresa", "indústria", "setor", "comércio"],
        "Justiça": ["justiça", "lei", "crime", "tribunal", "sentença"],
        "Sociedade": ["sociedade", "comunidade", "cidadão", "direitos", "igualdade"],
        "Transporte": ["transporte", "trânsito", "veículo", "estrada", "aeroporto"],
        "Moda": ["moda", "estilo", "tendência", "roupa", "acessório"],
        "Gastronomia": ["gastronomia", "culinária", "restaurante", "receita", "prato"],
    }

    categorized_news = defaultdict(list)

    for category, keywords in categories.items():
        if any(keyword.lower() in text.lower() for keyword in keywords):
            categorized_news[category].append(text)
            break 

    return categorized_news

def main(urls):
    news_by_category = defaultdict(list)

    for url in urls:
        text = extract_text_from_url(url)
        summary = summarize_text(text)
        categorized_news = categorize_news(summary)

        for category, news in categorized_news.items():
            news_by_category[category].extend(news)

    for category, news_list in news_by_category.items():
        if news_list:
            combined_text = ' '.join(news_list)
            final_summary = summarize_text(combined_text)
            print(f"Categoria: {category}")
            print(f"Resumo: {final_summary}\n")

urls = [
    "https://www.cisoadvisor.com.br/",
    "https://www.uol.com.br/",
    "https://g1.globo.com/",
    "https://valor.globo.com/",
]

if __name__ == "__main__":
    main(urls)
