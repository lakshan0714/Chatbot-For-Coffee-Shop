from .utils import get_llm_response,double_check_json_output
from dotenv import load_dotenv
import os
import json
from huggingface_hub import InferenceClient
from copy import deepcopy
load_dotenv()

class Input_classifier():
    def __init__(self):
        self.client=InferenceClient(api_key=os.getenv("INFERENCE_API"))
        self.model_name=os.getenv("MODEL")

    
    def get_response(self,messages):

        messages=deepcopy(messages)

        system_prompt = """
            You are a helpful AI assistant for a coffee shop application.
            Your task is to determine what agent should handle the user input. You have 3 agents to choose from:
            1. details_agent: This agent is responsible for answering questions about the coffee shop, like location, delivery places, working hours, details about menue items and price of items. Or listing items in the menu items. Or by asking what we have.
            2. order_taking_agent: This agent is responsible for taking orders from the user. It's responsible to have a conversation with the user about the order untill it's complete.
            3. recommendation_agent: This agent is responsible for giving recommendations to the user about what to buy. If the user asks for a recommendation, this agent should be used.

            Your output should be in a structured json format like so. each key is a string and each value is a string. Make sure to follow the format exactly:
            
            {
            "chain of thought": "go over each of the agents above and write some your thoughts about what agent is this input relevant to."
            "decision": "details_agent" or "order_taking_agent" or "recommendation_agent". Pick one of those. and only write the word.
            "message": leave the message empty.
            }
            dont use special characters inside or outside the strings specially / or \.

            Return only the JSON string. Do not include any extra text, explanations, or formatting outside the JSON.
            """


        input_messages=[{"role":"system","content":system_prompt}]+messages[-3:]

        chatbot_output=get_llm_response(self.client,input_messages,self.model_name,max_tokens=100)
       # print(chatbot_output)
       # print("\n\n")


       # print(self.postprocess(chatbot_output))
        verified_output=double_check_json_output(self.client,self.model_name,chatbot_output)
        #print(f"verified output from classification={ verified_output}")
       # print("\n\n")

        return self.postprocess(verified_output)
        

    def postprocess(self,output):

            out=json.loads(output)

            dict_output={

                "role":"assistant",
                "content":out['message'],
                "memory":{
                    "agent":"guard_agent",
                    "Classification_decision":out['decision']
                }
            }

            return dict_output
