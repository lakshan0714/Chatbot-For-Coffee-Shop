 AI Multi-Agent Chatbot with Recommendation Engine for Coffee Shop
 Built a recommendation engine using Apriori algorithms on sales data and embeddings of company private
 data, stored in Pinecone with structured JSON files for efficient storage and retrieval.
      Multi-Agents:
 Guard-Agent : Managed greetings and filtered coffee shop queries.
 Input Classifier Agent: Categorized user input into "Details Agent," "Recommendation Agent,"  and  
                                            "Order-Taking Agent."
 Details Agent: Retrieved private data using cosine-similarity from vector database(Pinecone) and reply to
 user.
 Recommendation Agent: Delivered suggestions using Apriori-based or popular-item recommendations.
 Order-Taking Agent: Tracked user orders and stored them in JSON format.
 Technologies: LLM (Llama 3.2), RAG, Prompt Engineering, MBA,NLP,Pinecone,Python,OOP,
                                     Huggingface,FastAPI,React Native.
