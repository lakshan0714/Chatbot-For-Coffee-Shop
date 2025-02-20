from .utils import get_llm_response,double_check_json_output
from dotenv import load_dotenv
import os
import json
from huggingface_hub import InferenceClient
from copy import deepcopy
load_dotenv()



class Guardagent():
    def __init__(self):
        self.client=InferenceClient(api_key=os.getenv("INFERENCE_API"))
        self.model_name=os.getenv("MODEL")

    
    def get_response(self,messages):

        messages=deepcopy(messages)

        system_prompt="""
        
         You are helpful AI assistant for a coffee shop application which serves drinks and pastries.
         Your task is to determine whether the user is asking something relevant to the coffee shop or not.
         
            The user is allowed to:
            1. Ask questions about the coffee shop, like location, working hours, menue items and coffee shop related questions.
            2. Ask questions about menue items, they can ask for ingredients in an item and more details about the item.
            3. Make an order.
            4. ASk about recommendations of what to buy.

            The user is NOT allowed to:
            1. Ask questions about anything else other than our coffee shop.
            2. Ask questions about the staff or how to make a certain menue item.

            and user allowed ask genral greetings and general words like hi? how are you? okay good bye

               Your output should be in a structured json format like so. each key is a string and each value is a string. Make sure to follow the format exactly:
            {
            "chain of Thought": "go over each of the points above and make see if the message lies under this point or not. Then you write some your thoughts about what point is this input relevant to."
            "decision": "allowed" , "not allowed" and "common greetings". Pick one of those. and only write the word.
            "message": leave the message empty if it's allowed,and respond to common greetings, else if not allowed write "Sorry, I can't help with that. Can I help you with your order?" and if it
            is common greetings answer polietly.
            }

        """


        input_messages=[{"role":"system","content":system_prompt}]+messages[-3:]

        chatbot_output=get_llm_response(self.client,input_messages,self.model_name,max_tokens=100)
        #print("Guard_bot_output:",chatbot_output)
       # print("\n\n")


       # print(self.postprocess(chatbot_output))
        verified_output=double_check_json_output(self.client,self.model_name,chatbot_output)
        #print(f"verified output={ verified_output}")
        #print("\n\n")

        return self.postprocess(verified_output)
        

    def postprocess(self,output):

            out=json.loads(output)

            dict_output={

                "role":"assistant",
                "content":out['message'],
                "memory":{
                    "agent":"guard_agent",
                    "guard_decision":out['decision']
                }
            }

            return dict_output
