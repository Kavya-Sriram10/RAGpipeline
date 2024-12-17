import requests
from bs4 import BeautifulSoup
import openai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Initialize OpenAI API key for GPT-3/4
openai.api_key = 'YOUR_OPENAI_API_KEY'

def scrape_website(url):
    
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    paragraphs = soup.find_all('p')
    text = " ".join([para.get_text() for para in paragraphs])
    return text

def generate_embeddings(text):
    
    response = openai.embeddings.create(
        model="text-embedding-ada-002",  # Specify the embedding model
        input=text  # Provide the text input for embedding
    )
    print(response)  # Debugging: Print the API response
    embeddings = [embedding['embedding'] for embedding in response['data']]
    return embeddings


def query_to_embeddings(query):
    
    response = openai.Embedding.create(
        input=query,
        model="text-embedding-ada-002"
    )
    return np.array(response['data'][0]['embedding']).reshape(1, -1)

def search_query(embeddings, query_embeddings, k=5):
    
    similarities = cosine_similarity(query_embeddings, embeddings)
    sorted_indices = np.argsort(similarities[0])[::-1]  # Sort indices by similarity
    return sorted_indices[:k]

def generate_response(query, retrieved_chunks):
    
    context = "\n".join(retrieved_chunks)
    prompt = f"Answer the following question based on the context below:\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"
    
    response = openai.Completion.create(
        engine="gpt-4",  # Using GPT-4 engine
        prompt=prompt,
        max_tokens=200
    )
    return response.choices[0].text.strip()

def handle_query(url, query):
    
    
    text = scrape_website(url)
    
    
    chunks = text.split('\n')  
    
   
    embeddings = generate_embeddings(chunks)

    
    query_embeddings = query_to_embeddings(query)
    
    
    indices = search_query(embeddings, query_embeddings)
    
    
    retrieved_chunks = [chunks[i] for i in indices]
    
    
    response = generate_response(query, retrieved_chunks)
    return response


def run_backend():
    print("Welcome to the RAG Query System!")
    
    while True:
        r
        url = input("Enter the website URL (or 'exit' to quit): ").strip()
        if url.lower() == 'exit':
            break
        
        query = input("Enter your question: ").strip()
        
        
        response = handle_query(url, query)
        
        print("\nGenerated Response:")
        print(response)
        print("\n" + "-"*50)


if __name__ == "__main__":
    run_backend()
