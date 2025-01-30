LlamaIndex, is a data framework which has capabilities to load, ingest, store, query data. We can build data pipelines to feed relevant data to LLMs and build applications around them. In this series, RAG is the main application covered in this series. This series specifically covers developing production grade RAG application.

LlamaIndex has a seperate python package for each individual components. Ex OllamaEmbedding, Ollamallm requires (llama-index-llms-ollama llama-index-embeddings-ollama packages respectivley.
# Fundamental Components in LlamaIndex

1. ___Vector Stores___: This is a database to store high dimensional data like text, image, audio etc in their vector representation(numbers).
	* They don't perform keyword search like BoW, TF-IDF etc. They search semantically relevant documents or vectors in-accordance with user query.
	* Similarity Search is the primary function and is crucial in many AI Systems like recommendation engine, image retrieval platforms.
	* Similarity search finds semantically relevant vectors based on user query.
2. ___Data Connectors___: Data Connectors or Readers are available in LlamaHub(integrations are available for PDF, APIs, SQL etc).
	* RAG applications performance depends on the various data sources integrated into vector stores. Readers does this in LlamaIndex.
	* Automates task of fetching data from various sources.
	* Processes data into _Document_ objects.
3. ___Nodes___: Nodes are smaller converted units from _Document_ object.
4. ___Indices___: Indices is the heart of LlamaIndex. Creation of index from unstructured data making it searchable. LlamaIndex offers a variety of indices:
	* _SummaryIndex_, Creates a summary and stores it with all nodes in the index for easier querying against summary node instead of all smaller nodes.
	* _VectorStoreIndex_, Creates embeddings for documents and stores them in a specified vector store.
5. ___Query Engines___: Is a wrapper around IndexStores(Indices). We can use them as retrieve and generation(as_query_engine) or use them as retrievers alone for customized usage. This is a sophisticated interface to interact with data using natural language
6. ___Routers___: Decides most appropriate retriever to extract content from knowledge base. In a case where two data sources(sql, vector source), router determines the optimal retriever for task in hand.

# Prompt vs Fine-Tuning vs RAG

Source: https://learn.activeloop.ai/courses/take/rag/multimedia/51320353-brief-overview-of-available-techniques-fine-tuning-rag-activeloop-s-deep-memory

## Deep Lake Deep Memory

Deep Memory is a fine-tuning technique to train a model for question answer pairs to obtain a embedding model relevant to the domain or task at hand. This improves the retrieval accuracy(recall) for better text generation with RAG.

## Deep Memory Steps

This technique will be useful in general.

1. Create Question Answer Pairs. Use an LLM to generate questions using context. Now create pairs, use an ID for context to maintain the dataset.
2. Train an embedding model on these pairs. Now we've a model more tune for this specific context.
3. Use this embedding model to transform context and question queries. Now the vector world is more attuned for this domain.
4. Now vector search will return much better retrieval recall results.

## Synthetic QA generation

```Python
from openai import OpenAI
client = OpenAI()

def generate_question(text):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a world class expert for generating questions based on provided context. \
                        You make sure the question can be answered by the text."},
                {
                    "role": "user",
                    "content": text,
                },
            ],
        )
        return response.choices[0].message.content
    except:
        question_string = "No question generated"
        return question_string
```

[Using Deep Memory to evaluate Retrieval with Deep Lake vector store](https://learn.activeloop.ai/courses/take/rag/multimedia/51320354-using-deep-memory-to-boost-retrieval-accuracy-up-to-22-on-average)