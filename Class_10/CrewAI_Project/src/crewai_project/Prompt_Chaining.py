from crewai.flow.flow import Flow,start,listen
from litellm import completion
import os
from dotenv import load_dotenv

load_dotenv()

class SimpleFlow(Flow,tracing=False):
    
    @start()
    def generate_random_city(self):
        response=completion(
            model="gemini/gemini-2.5-flash",
            api_key=os.getenv("GEMINI_API_KEY"),
            messages=[
                {
                    "role": "user",
                    "content": "Generate a random city name of Pakistan."
                }
            ]
        )
        city=response.choices[0].message.content
        print(f"\nGenerated City: {city}")
        return city # Mai next function ko ye city pass kr rha hon es sy sirf aik hi function mai access ho gii jo es kay run honay ka wait kr rha ha
        # self.state['city']=city es tarha b pass kr sakta tha agr es tarha krta tu wo hr function mai access ho jata


    @listen(generate_random_city)
    def generate_fact(self,city):
        response=completion(
            model="gemini/gemini-2.5-flash",
            api_key=os.getenv("GEMINI_API_KEY"),
            messages=[
                {
                    "role": "user",
                    "content": f"Generate a 3 facts about this {city} city"
                }
            ]
        )
        fact=response.choices[0].message.content
        print(f"\nGenerated Facts:\n{fact}")
        self.state['fact']=fact
    
    
    @listen(generate_fact)
    def save_fact(self):
        with open("Fact.md","w") as file:
            file.write(self.state['fact'])
        
        print("\nFact saved successfully.")
        return self.state['fact']


def start_func():
    obj=SimpleFlow()
    obj.kickoff()