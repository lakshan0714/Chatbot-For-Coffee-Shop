from agents import Guardagent
import os
from agents import Input_classifier,Detailsagent,Recommendation_agent


def main():
   guard_agent=Guardagent()
   Input_classification=Input_classifier()
   Details_agent=Detailsagent()
   recommendation_agent=Recommendation_agent('recomendation_objects/apriori_recommendations.json',
                                                    'recomendation_objects/popularity_recommendation.csv')
   
   agent_dict={
      "details_agent":Details_agent,
   #   "order_taking_agent" :
      "recommendation_agent":recommendation_agent,
      
      
   }
   guard_agent=Guardagent()
   Input_classification=Input_classifier()


   messages=[]
   while True:
     #os.system('cls' if os.name == 'nt' else 'clear')

     print("\n\n Print Messages...........")
     for message in messages:
        print(f"{message['role']}:{message['content']}")
    
    #Get user Input

     prompt=input("User:")
     messages.append({"role":"user","content":prompt})

     guard_agent_response=guard_agent.get_response(messages)

     if guard_agent_response['memory']['guard_decision']=='not allowed' or guard_agent_response['memory']['guard_decision']=='common greetings':
        messages.append(guard_agent_response)
        continue

     Choosen_agent=Input_classification.get_response(messages)["memory"]["Classification_decision"]

     print("choosen Agent:",Choosen_agent)

     agent=agent_dict[Choosen_agent]
     response=agent.get_response(messages)
     messages.append(response)



if __name__ == "__main__":
    main()