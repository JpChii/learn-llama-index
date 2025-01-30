Retrieval is the main bottleneck in RAG. Techniques like query expansion, query transformations and query construction and strategies like Reranking, Recursive retrieval and small to big retrieval improves the retrieval process.

Querying in LlamaIndex works with Retrievers, Query Engine, Query Transform and relevancy of retrieval can be improved with above techniques.

# Query Construction

Various cookbooks, docs to create natural language queries to retrieve information from structured, structured + unstructured and unstructured databases. https://blog.langchain.dev/query-construction/

Query Construction is taking the natural language query and converting it into the query language of the database. In general for unstructured data this is conversion of user query to vector. But unstructured vector stores also often accompanied with structured metadata(Ex: Elastic Search)

Below are some of the major Query construction requirements:
1. Text-to-metadata-filter
2. Text-to-SQL
3. Text-to-SQL+Semantic
4. Text-to-Cypher

We can infer logical filters from natural language and then use it for filtering the vector database instead of searching its entirety or we can refine the retrieved chunks before passing it to LLM for generation.

# Query Expansion

Query expansion is extending the original query with additional terms of phrases that are synonymous. We can do this with `synonym_expand_policy` from `KnowledgeGraphRAGRetriever` In LlamaIndex this is enhanced when combined with QueryTransformation.

# Query Transformation

Query transformation changes query structure to improve it's performance. Ex query: What's Microsoft revenue in 2021? to Microsoft revenues 2021. This involves changing structure, adding synonyms, adding context.

[More detailed study](https://blog.langchain.dev/query-transformations/)

# Sub Question Query Engine

We can combine multiple query engines to cater for complex data interrogation needs.

We'll use a query engine as a tool and create a SubQuestionQueryEngine to generate multiple queries from single query. Then run retrieval on multiple queries and combine the results.

With this Retriever, we get better results. Creation of custom Retriever Engines nuanced for individual distinct queries is possible in llamaIndex

# QueryRetrieverEngine

Custom Retrievers are combination of different retrieval techniques(BM25, Sparse, Dense, Vector) and fusion of retrieved results at the end. The retrieve is an important piece as it's retrieves the relevant context(not available to LLM) to generate better responses for the user query.

There are two types:
1. ___VectorIndexRetriever:___ Retrieves _top_k_ results based on relevance, similarity. Ensures results aligns with query's intent.
2. ___SummaryIndexRetriever___: Retrieves all nodes related to user query without prioritizing their relevance. This is less concerned about aligning with context and to provide a more broader view.

We can create a CustomFusionRetriever from scratch using llama index. [Detailed guide](https://docs.llamaindex.ai/en/stable/examples/low_level/fusion_retriever/#step-1-query-generationrewriting)

This FusionRetriever performs below:
1. Generate multiple queries
2. Use multiple retrievers(BM25(dense), full text search, vector search) 
3. Combine or fuse the results with RRF(reciprocal rank fusion).

This FusionRetriever can be plugged into `RetrieverQueryEngine` to create a CustomRetrieverEngine.

# Reranking

* Reranking retrieved results based on relevancy with query.
* This begins with batching retrieved documents and assigning a score to these batches using an LLM with respect to the query.
* Sort the results based on this score.
* Reranking is a improvement step post retrieval. It can improve the performance without overhaul of the entire system.
* Cost-effective solution to improving search functionality.

Cohere Reranker is one of the libraries to perform this injunction with a query engine. 

```Python
import os
from llama_index.postprocessor.cohere_rerank import CohereRerank

cohere_rerank = CohereRerank(api_key=os.environ['COHERE_API_KEY'], top_n=2)

query_engine = vector_index.as_query_engine(
    similarity_top_k=10,
    node_postprocessors=[cohere_rerank],
)

response = query_engine.query(
    "What did Sam Altman do in this essay?",
)
print(response)
```

# Reference materials

1. For advanced retrieval Recursive Retrieval, Big to Small Retrieval from here https://learn.activeloop.ai/courses/take/rag/multimedia/51334510-mastering-advanced-rag-techniques-with-llamaindex.
2. Check Retrievers section in https://docs.llamaindex.ai/en/stable/examples
3. Overview and classification of techniques for different QA points - https://learn.activeloop.ai/courses/take/rag/multimedia/51625719-advanced-retrieval-strategies-deep-memory-small-to-big-retrieval-structured-and-unstructured-data-parsing-tables-and-mroe
