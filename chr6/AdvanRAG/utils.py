import faiss
import numpy as np
from elasticsearch import Elasticsearch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BertTokenizer, BertForSequenceClassification
from sentence_transformers import SentenceTransformer
import chromadb
import os
import json

from elasticsearch import Elasticsearch, ConnectionError
from langchain_elasticsearch import ElasticsearchStore
from langchain_groq import ChatGroq
import torch.nn.functional as F


# Load the pre-trained models and tokenizers for text generation, sentence embedding,
# and reranking.

# Load the SentenceTransformer model for encoding queries and documents
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')  # Small, fast model for embeddings


# Load configuration from config.txt file
def load_config(config_path='config.txt'):
    """Load API keys from config file."""
    with open(config_path, 'r') as f:
        return json.load(f)

# Load configuration globally
config = load_config()

# Helper function to get the language model
def get_llm():
    """
    Returns the language model instance.

    This function initializes and returns a ChatGroq language model configured with the specified model name,
    temperature, maximum tokens, and other settings.

    Returns:
        ChatGroq: An instance of the ChatGroq language model.
    """
    # Set environment variable
    os.environ["GROQ_API_KEY"] = config.get('GROQ_API_KEY')
    
    llm = ChatGroq(
        model="meta-llama/llama-4-maverick-17b-128e-instruct",
        temperature=0,
        max_tokens=1024,
    )
    return llm


def advanced_query_transformation(query):
    """
    Transforms the input query by adding synonyms, extensions, or modifying the structure
    for better search performance.

    Args:
        query (str): The original query.

    Returns:
        str: The transformed query with added synonyms or related terms.
    """
    # Example transformation: adding an OR clause with a related term
    expanded_query = query + " OR related_term"
    return expanded_query

def advanced_query_routing(query):
    """
    Determines the retrieval method based on the presence of specific keywords in the query.

    Args:
        query (str): The user's query.

    Returns:
        str: 'textual' if the query requires text-based retrieval, 'vector' otherwise.
    """
    if "specific_keyword" in query:
        return "textual"
    else:
        return "vector"

# Fusion Retrieval Function
def fusion_retrieval(query, collection, documents, es, index_name='movies', top_k=5):
    """
    Retrieves the top_k most relevant documents using a combination of vector-based
    and textual retrieval methods.

    Args:
        query (str): The search query.
        collection: ChromaDB collection
        documents: List of documents
        es: Elasticsearch client
        index_name (str): Name of the Elasticsearch index
        top_k (int): The number of top documents to retrieve.

    Returns:
        list: A list of combined results from both vector and textual retrieval methods.
    """
    # Vector-based retrieval using sentence embeddings
    query_embedding = sentence_model.encode(query).tolist()
    vector_results = collection.query(query_embeddings=[query_embedding], n_results=min(top_k, len(documents)))

    # Extract documents from ChromaDB metadata
    # ChromaDB stores text in metadatas when only embeddings are provided
    chroma_documents = []
    if vector_results['metadatas'] and vector_results['metadatas'][0]:
        for metadata in vector_results['metadatas'][0]:
            if metadata and 'chunk' in metadata:
                chroma_documents.append(metadata['chunk'])
    
    # Fallback: if documents field is populated, use it
    if not chroma_documents and vector_results['documents'] and vector_results['documents'][0]:
        chroma_documents = [doc for doc in vector_results['documents'][0] if doc]

    # Textual retrieval using Elasticsearch (updated syntax without 'body' parameter)
    es_results = es.search(
        index=index_name,
        size=top_k,
        query={
            "match": {
                "content": query
            }
        }
    )
    es_documents = [hit["_source"]["content"] for hit in es_results['hits']['hits']]
    
    print(f"ChromaDB Documents: {len(chroma_documents)} retrieved")
    print(f"Elasticsearch Documents: {len(es_documents)} retrieved")
    
    # Combine results from both retrieval methods
    combined_results = chroma_documents + es_documents

    return combined_results

import torch.nn.functional as F

# Document Reranking Function
def rerank_documents(query, documents, rerank_tokenizer, rerank_model):
    """
    Reranks the retrieved documents based on their relevance to the query using a pre-trained
    BERT model.

    Args:
        query (str): The user's query.
        documents (list): A list of documents retrieved from the search.
        rerank_tokenizer: BERT tokenizer
        rerank_model: BERT model for reranking

    Returns:
        list: A list of reranked documents, sorted by relevance.
    """
    # Filter out None and empty documents
    valid_documents = [doc for doc in documents if doc and isinstance(doc, str) and doc.strip()]
    
    if not valid_documents:
        print("Warning: No valid documents to rerank")
        return []
    
    inputs = [rerank_tokenizer.encode_plus(query, doc, return_tensors='pt', truncation=True, padding=True) for doc in valid_documents]

    # Use logits to get scores
    scores = []
    for input in inputs:
        outputs = rerank_model(**input)
        logits = outputs.logits
        probabilities = F.softmax(logits, dim=1)
        positive_class_probability = probabilities[:, 1].item()  # Assuming the second element represents the positive class
        scores.append(positive_class_probability)

    ranked_docs = sorted(zip(valid_documents, scores), key=lambda x: x[1], reverse=True)
    print("Ranked Docs:", ranked_docs)
    return [doc for doc, score in ranked_docs]

def select_and_compress_context(documents, summarizer):
    """
    Summarizes the content of the retrieved documents to create a compressed context.

    Args:
        documents (list): A list of documents to summarize.
        summarizer: Summarization pipeline

    Returns:
        list: A list of summarized texts for each document.
    """
    print("documents to summarize:", documents)
    summarized_context = []
    for doc in documents:
        # Skip None or empty documents
        if doc is None or not isinstance(doc, str) or not doc.strip():
            continue
            
        input_length = len(doc.split())  # Calculate input length based on word count
        
        # Skip very short documents that can't be summarized
        if input_length < 10:
            summarized_context.append(doc)
            continue
            
        max_length = min(100, input_length - 1)  # Set max_length to input_length if smaller than 100
        min_length = min(5, input_length - 1)
        
        try:
            summary = summarizer(doc, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']
            summarized_context.append(summary)
        except Exception as e:
            print(f"Warning: Could not summarize document, using original: {str(e)[:100]}")
            summarized_context.append(doc)
    
    return summarized_context

# Answer Generation Function
def generate_answer(query, chunks, llm):
    """
    Generates an answer based on the input query and context chunks using a language model.

    Args:
        query (str): The user's query.
        chunks (list): A list of context chunks to inform the answer.
        llm (ChatGroq): An instance of the ChatGroq language model.

    Returns:
        str: The generated answer.
    """
    # Combine chunks into a single context string
    context = "\n\n".join(chunks)

    # Construct the prompt for the language model as a string
    prompt = f"""[INST]
Instruction: You're an expert in movie suggestions. Your task is to analyze carefully the context and come up with an exhaustive answer to the following question:
{query}

Here is the context to help you:

{context}

[/INST]"""

    # Invoke the language model with the prompt
    response = llm.invoke(prompt)  # Pass the prompt as a string directly

    # Since response is likely an AIMessage object, access the content directly
    generated_text = response.content

    return generated_text


