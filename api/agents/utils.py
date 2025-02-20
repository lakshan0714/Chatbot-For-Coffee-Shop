"""
store more common functions
"""

from langchain_community.embeddings import HuggingFaceBgeEmbeddings
import re

'''
model_name = "BAAI/bge-small-en"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}

'''

def get_llm_response(client,messages,model,max_tokens):
    client=client
    
    completion = client.chat.completions.create(
    model=model, 
	messages=messages, 
	max_tokens=max_tokens,
   
   # Temprature=temprature
)
    return completion.choices[0].message.content



def get_embedding(model_name,model_kwargs,encode_kwargs,prompt):
    hf = HuggingFaceBgeEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)
    embed=hf.embed_query(prompt)

    return embed



def double_check_json_output(client,model_name,json_string):

    prompt = f"""
You will validate a JSON string based on the following rules:

1. If the string contains invalid JSON, correct any issues to make it valid.
2. If any curly braces are missing in the JSON, add the necessary braces to complete it.
3. Ensure the JSON object contains only three keys: "chain of thought", "decision", and "message".
   - If the keys are named differently, rename them to match the specified order exactly.
4. Do not add any additional characters, such as `/` or `\`, inside or outside the JSON string. Retain only the curly braces `{{}}` for structure.
5. If the JSON is already correct, return it as is without making changes.
6. Your response must strictly contain only the final JSON string.
7. Do not include any extra text, explanations, or formatting outside the JSON string.

Process the provided JSON string accordingly:

{json_string}
"""


    messages = [{"role": "system", "content": prompt}]

    response = get_llm_response(client,messages,model_name,max_tokens=100)
    

    return response





