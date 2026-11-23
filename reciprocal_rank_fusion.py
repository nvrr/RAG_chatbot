from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from pydantic import BaseModel
from typing import List
from collections import defaultdict

load_dotenv()

# Setup
persistent_directory = "db/chroma_db"
# embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",

    model_kwargs={
        "device": "cpu"
    },

    encode_kwargs={
        "normalize_embeddings": True,
        "batch_size": 2
    }
)

# llm = ChatOpenAI(model="gpt-4o", temperature=0)

llm = ChatOllama(
            model="qwen3:0.6b",
            base_url="http://localhost:11434",
            temperature=0
        )

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)

# Pydantic model for structured output
class QueryVariations(BaseModel):
    queries: List[str]
# ──────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ──────────────────────────────────────────────────────────────────

# Original query
original_query = "How does Tesla make money?"
print(f"Original Query: {original_query}\n")

# ──────────────────────────────────────────────────────────────────
# Step 1: Generate Multiple Query Variations
# ──────────────────────────────────────────────────────────────────

llm_with_tools = llm.with_structured_output(QueryVariations)

prompt = f"""Generate 3 different variations of this query that would help retrieve relevant documents:

Original query: {original_query}

Return 3 alternative queries that rephrase or approach the same question from different angles."""

response = llm_with_tools.invoke(prompt)
query_variations = response.queries

print("Generated Query Variations:")
for i, variation in enumerate(query_variations, 1):
    print(f"{i}. {variation}")

print("\n" + "="*60)

# ──────────────────────────────────────────────────────────────────
# Step 2: Search with Each Query Variation & Store Results
# ──────────────────────────────────────────────────────────────────

retriever = db.as_retriever(search_kwargs={"k": 5})  # Get more docs for better RRF
all_retrieval_results = []  # Store all results for RRF

for i, query in enumerate(query_variations, 1):
    print(f"\n=== RESULTS FOR QUERY {i}: {query} ===")
    
    docs = retriever.invoke(query)
    all_retrieval_results.append(docs)  # Store for RRF calculation
    
    print(f"Retrieved {len(docs)} documents:\n")
    
    for j, doc in enumerate(docs, 1):
        print(f"Document {j}:")
        print(f"{doc.page_content[:150]}...\n")
    
    print("-" * 50)

print("\n" + "="*60)
print("Multi-Query Retrieval Complete!")
print("Notice how different query variations retrieved different documents.")

# ──────────────────────────────────────────────────────────────────
# Step 3: Apply Reciprocal Rank Fusion (RRF)
# ──────────────────────────────────────────────────────────────────

def reciprocal_rank_fusion(chunk_lists, k=60, verbose=True):

    if verbose:
        print("\n" + "="*60)
        print("APPLYING RECIPROCAL RANK FUSION")
        print("="*60)
        print(f"\nUsing k={k}")
        print("Calculating RRF scores...\n")
    
    # Data structures for RRF calculation
    rrf_scores = defaultdict(float)  # Will store: {chunk_content: rrf_score}
    all_unique_chunks = {}  # Will store: {chunk_content: actual_chunk_object}
    
    # For verbose output - track chunk IDs
    chunk_id_map = {}
    chunk_counter = 1
    
    # Go through each retrieval result
    for query_idx, chunks in enumerate(chunk_lists, 1):
        if verbose:
            print(f"Processing Query {query_idx} results:")
        
        # Go through each chunk in this query's results
        for position, chunk in enumerate(chunks, 1):  # position is 1-indexed
            # Use chunk content as unique identifier
            chunk_content = chunk.page_content
            
            # Assign a simple ID if we haven't seen this chunk before  , ex: "Tesla vehicle sales": "Chunk_1",
            if chunk_content not in chunk_id_map:
                chunk_id_map[chunk_content] = f"Chunk_{chunk_counter}"
                chunk_counter += 1

            #it returns chunk id that we assign above like: chunk_1            
            chunk_id = chunk_id_map[chunk_content] 

            # Store the chunk object (in case we haven't seen it before)
            #{"Tesla generates revenue through electric vehicle sales.": Document(
            #  page_content="Tesla generates revenue through electric vehicle sales.",
            #         metadata={"source": "tesla.pdf", "page": 1}  )}
            all_unique_chunks[chunk_content] = chunk
            
            # Calculate position score: 1/(k + position)
            position_score = 1 / (k + position)

            #defaultdict(float) initializes its score to 0.0
            # Add to RRF score :0.0 + 0.01639 = 0.01639, { "Tesla generates revenue through electric vehicle sales.": 0.01639}
            rrf_scores[chunk_content] += position_score
            
            if verbose:
                print(f"  Position {position}: {chunk_id} +{position_score:.4f} (running total: {rrf_scores[chunk_content]:.4f})")
                print(f"    Preview: {chunk_content[:80]}...")
        
        if verbose:
            print()
    
    # Sort chunks by RRF score (highest first)
    sorted_chunks = sorted(
        [(all_unique_chunks[chunk_content], score) for chunk_content, score in rrf_scores.items()],
        key=lambda x: x[1],  # Sort by RRF score
        reverse=True  # Highest scores first
    )

    #explanation:

#     rrf_scores = {
#     "Tesla makes money from vehicle sales": 0.048,
#     "Tesla sells battery storage products": 0.032,
#     "Tesla generates revenue from services": 0.025
# }
# rrf_scores.items():   Returns key-value pairs:
# [
#     ("Tesla makes money from vehicle sales", 0.048),
#     ("Tesla sells battery storage products", 0.032),
#     ("Tesla generates revenue from services", 0.025)
# ]

# all_unique_chunks = {
#     "Tesla makes money from vehicle sales": Document_A,
#     "Tesla sells battery storage products": Document_B,
#     "Tesla generates revenue from services": Document_C
# }

# (all_unique_chunks[chunk_content], score)
# Creates a tuple containing:
# 1. The original Document object.
# 2. Its RRF score.
# [
#     (Document_A, 0.025),
#     (Document_B, 0.048),
#     (Document_C, 0.032)
# ]

# Understand key=lambda x: x[1]
# This tells Python which value it should use for sorting.

# Each item is a tuple:
# (Document_A, 0.048)



# Python indexes:
# x[0]  # Document objectx[1]  # RRF score



# Therefore:
# key=lambda x: x[1]



# means sort using the second element of each tuple (the RRF score).
# 5. Understand reverse=True
# reverse=True



# Sorts in descending order, meaning highest score first.

# For example:

# Before sorting:
# [    (Document_A, 0.025),    (Document_B, 0.048),    (Document_C, 0.032)]



# After sorting:
# [    (Document_B, 0.048),    (Document_C, 0.032),    (Document_A, 0.025)]



# Final result
# sorted_chunks contains the original Document objects paired with their RRF scores, ordered from most relevant (highest score) to least relevant (lowest score).

 
    if verbose:
        print(f"✅ RRF Complete! Processed {len(sorted_chunks)} unique chunks from {len(chunk_lists)} queries.")
    
    return sorted_chunks

# Apply RRF to our retrieval results
fused_results = reciprocal_rank_fusion(all_retrieval_results, k=60, verbose=True)

# ──────────────────────────────────────────────────────────────────
# Step 4: Display Final Fused Results
# ──────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("FINAL RRF RANKING")
print("="*60)

print(f"\nTop {min(10, len(fused_results))} documents after RRF fusion:\n")

for rank, (doc, rrf_score) in enumerate(fused_results[:10], 1):
    print(f"🏆 RANK {rank} (RRF Score: {rrf_score:.4f})")
    print(f"{doc.page_content[:200]}...")
    print("-" * 50)

print(f"\n✅ RRF Complete! Fused {len(fused_results)} unique documents from {len(query_variations)} query variations.")
print("\n💡 Key benefits:")
print("   • Documents appearing in multiple queries get boosted scores")
print("   • Higher positions contribute more to the final score") 
print("   • Balanced fusion using k=60 for gentle position penalties")

# ──────────────────────────────────────────────────────────────────
# Optional: Quick Usage Examples
# ──────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("USAGE EXAMPLES")
print("="*60)























# Your original code:
# sorted_chunks = sorted(    chunks,    key=lambda x: x[1],    reverse=True)



# 1. Using a normal function (alternative)
# def get_score(item):    return item[1]sorted_chunks = sorted(    chunks,    key=get_score,    reverse=True)



# Here, get_score() receives each tuple and returns its second element (the RRF score).
# 2. Using operator.itemgetter() (clean alternative)
# from operator import itemgettersorted_chunks = sorted(    chunks,    key=itemgetter(1),    reverse=True)



# itemgetter(1) extracts the second element of each tuple.
# Comparison
# Method	Code
# Lambda	key=lambda x: x[1]
# Normal function	key=get_score
# Itemgetter	key=itemgetter(1)