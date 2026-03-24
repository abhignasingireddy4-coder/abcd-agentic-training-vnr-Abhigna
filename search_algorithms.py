import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ─────────────────────────────────────────
# 1. COSINE SIMILARITY SEARCH
# ─────────────────────────────────────────
def cosine_search(query: str, documents: list) -> list:
    """
    Finds the most similar documents to the query using Cosine Similarity.
    Uses TF-IDF vectors to represent text.
    """
    corpus = [query] + documents
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)

    query_vec = tfidf_matrix[0]
    doc_vecs = tfidf_matrix[1:]

    scores = cosine_similarity(query_vec, doc_vecs).flatten()
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

    print("\n🔍 Cosine Similarity Search Results:")
    results = []
    for rank, (idx, score) in enumerate(ranked, 1):
        print(f"  Rank {rank}: [{score:.4f}] {documents[idx]}")
        results.append((documents[idx], score))
    return results


# ─────────────────────────────────────────
# 2. TF-IDF RANKED SEARCH
# ─────────────────────────────────────────
def tfidf_search(query: str, documents: list) -> list:
    """
    Ranks documents by TF-IDF score against the query.
    """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    query_vec = vectorizer.transform([query])

    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

    print("\n🔍 TF-IDF Search Results:")
    results = []
    for rank, (idx, score) in enumerate(ranked, 1):
        print(f"  Rank {rank}: [{score:.4f}] {documents[idx]}")
        results.append((documents[idx], score))
    return results


# ─────────────────────────────────────────
# 3. JACCARD SIMILARITY SEARCH
# ─────────────────────────────────────────
def jaccard_search(query: str, documents: list) -> list:
    """
    Ranks documents by Jaccard similarity (word overlap) with the query.
    """
    def jaccard(set_a, set_b):
        intersection = set_a & set_b
        union = set_a | set_b
        return len(intersection) / len(union) if union else 0.0

    query_words = set(query.lower().split())
    scores = [(doc, jaccard(query_words, set(doc.lower().split()))) for doc in documents]
    ranked = sorted(scores, key=lambda x: x[1], reverse=True)

    print("\n🔍 Jaccard Similarity Search Results:")
    for rank, (doc, score) in enumerate(ranked, 1):
        print(f"  Rank {rank}: [{score:.4f}] {doc}")
    return ranked


# ─────────────────────────────────────────
# 4. LINEAR SEARCH
# ─────────────────────────────────────────
def linear_search(query: str, documents: list) -> list:
    """
    Simple linear scan — returns indices of documents containing the query keyword.
    O(n) time complexity.
    """
    print("\n🔍 Linear Search Results:")
    results = []
    for idx, doc in enumerate(documents):
        if query.lower() in doc.lower():
            print(f"  Found at index {idx}: {doc}")
            results.append(idx)
    if not results:
        print("  No matches found.")
    return results


# ─────────────────────────────────────────
# 5. BINARY SEARCH
# ─────────────────────────────────────────
def binary_search(sorted_docs: list, target: str) -> int:
    """
    Binary search on a sorted list of documents.
    Returns the index if found, else -1.
    O(log n) time complexity.
    """
    low, high = 0, len(sorted_docs) - 1
    target = target.lower()

    print("\n🔍 Binary Search Result:")
    while low <= high:
        mid = (low + high) // 2
        mid_val = sorted_docs[mid].lower()
        if mid_val == target:
            print(f"  Found at index {mid}: {sorted_docs[mid]}")
            return mid
        elif mid_val < target:
            low = mid + 1
        else:
            high = mid - 1

    print(f"  '{target}' not found.")
    return -1


# ─────────────────────────────────────────
# MAIN - DEMO
# ─────────────────────────────────────────
if __name__ == "__main__":
    documents = [
        "Python is a popular programming language for AI",
        "Machine learning uses algorithms to learn from data",
        "Deep learning is a subset of machine learning",
        "Natural language processing deals with text and speech",
        "Cosine similarity measures the angle between two vectors",
        "Search algorithms help retrieve relevant information quickly",
        "Agentic AI systems can plan and execute multi-step tasks",
        "Vector databases store embeddings for fast similarity search",
    ]

    query = "machine learning and AI search"

    print("=" * 55)
    print(f"  Query: '{query}'")
    print("=" * 55)

    cosine_search(query, documents)
    tfidf_search(query, documents)
    jaccard_search(query, documents)
    linear_search("machine learning", documents)

    sorted_docs = sorted(documents)
    binary_search(sorted_docs, "Natural language processing deals with text and speech")
```

3. Scroll down to **"Commit changes"**
4. In the commit message box type:
```
Add search algorithms implementation
```
5. Click **"Commit changes"** ✅

---

### Step 3: Create `requirements.txt`

1. Click **"Add file"** → **"Create new file"** again
2. In filename box type:
```
requirements.txt
```
3. In the text area paste:
```
numpy
scikit-learn
```
4. Commit message:
```
Add requirements.txt
