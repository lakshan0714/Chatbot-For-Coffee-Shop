from agents import Guardagent
import os
from agents import Input_classifier,Detailsagent,Recommendation_agent


class agentcontroller():
   
   def __init__(self):
    self.guard_agent=Guardagent()
    self.Input_classification=Input_classifier()
    self.Details_agent=Detailsagent()
    self.recommendation_agent=Recommendation_agent('recomendation_objects/apriori_recommendations.json',
                                                    'recomendation_objects/popularity_recommendation.csv')
   
    self.agent_dict={
      "details_agent":self.Details_agent,
   #   "order_taking_agent" :
      "recommendation_agent":self.recommendation_agent,
      
      
   }
   
   def get_response(self,input):
     job_input = input["input"]
     messages = job_input["messages"]

     guard_agent_response=self.guard_agent.get_response(messages)

     if guard_agent_response['memory']['guard_decision']=='not allowed' or guard_agent_response['memory']['guard_decision']=='common greetings':
        return guard_agent_response
        

     Choosen_agent=self.Input_classification.get_response(messages)["memory"]["Classification_decision"]
     agent=self.agent_dict[Choosen_agent]
     response=agent.get_response(messages)
     return response




