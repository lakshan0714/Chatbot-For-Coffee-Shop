from .utils import get_llm_response,get_embedding
from dotenv import load_dotenv
import os
import json
from huggingface_hub import InferenceClient
from copy import deepcopy
load_dotenv()
from pinecone import Pinecone, ServerlessSpec


class Detailsagent():
    def __init__(self):
        self.client=InferenceClient(api_key=os.getenv("INFERENCE_API"))
        self.model_name=os.getenv("MODEL")
        self.Pinecone_api_key=os.getenv("PINECONE_API")
        self.PINECONE_INDEX=os.getenv("PINECONE_INDEX")
        self.embed_model_name = "BAAI/bge-small-en"
        self.model_kwargs = {"device": "cpu"}
        self.encode_kwargs = {"normalize_embeddings": True}
        self.pc = Pinecone(api_key=self.Pinecone_api_key)

    def getting_closest_vectors(self,index_name,input_embeddings,top_k=2):
         index=self.pc.Index(index_name)

         results=index.query(
             namespace="ns1",
             vector=input_embeddings,
             top_k=top_k,
             include_metadata=True,
             include_values=False)
    
         return results
    

    def get_response(self,messages):
        messages=deepcopy(messages)

        user_message=messages[-1]['content']

        user_embedding=get_embedding(self.embed_model_name,self.model_kwargs,self.encode_kwargs,messages[-1]['content'])
        
        result=self.getting_closest_vectors(self.PINECONE_INDEX,user_embedding,top_k=1)
       # print(result)

        sorce_knowledge="\n" .join([x['metadata']['text'].strip()+'\n' for x in result['matches']])
        

        prompt=f"""
        using context below and answer the question.dont answer irelevent things apart from below resources

        {sorce_knowledge}
         
        Query:{user_message}
        """
        #print(prompt)

        messages[-1]['content']=prompt

        system_prompt="""You are a customer care agent for a coffee shop 
        You should answer to user assume your a waiter.respond based on theire input dont response irrelavnt things.
        """

        input_messages=[{"role":"system","content":system_prompt}]+messages[-3:]

        chatbot_output=get_llm_response(self.client,input_messages,self.model_name,max_tokens=100)
       # print(chatbot_output)
       # print("\n\n")


       

        return self.postprocess(chatbot_output)
        

    def postprocess(self,output):

           

            dict_output={

                "role":"assistant",
                "content":output,
                "memory":{
                    "agent":"Details_agent",
                    
                }
            }

            return dict_output


