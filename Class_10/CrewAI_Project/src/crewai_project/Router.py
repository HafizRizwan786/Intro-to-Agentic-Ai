from crewai.flow.flow import Flow, start, listen, router
from litellm import completion
from dotenv import load_dotenv
import os

load_dotenv()


class SimpleFlow(Flow):

    @start()
    def greeting(self):
        print("Asslam-o-Alikum")


    @router(greeting)
    def select_city(self):
        response = completion(
            model="gemini/gemini-2.5-flash",
            api_key=os.getenv("GEMINI_API_KEY"),
            messages=[
                {
                    "role": "user",
                    "content": """
                    Choose ONLY ONE city from the following list.

                    Faisalabad
                    Lahore
                    Karachi

                    Return ONLY the city name.
                    Do not return any explanation.
                    """
                }
            ]
        )

        city = response.choices[0].message.content.strip()
        print(f"\nSelected City: {city}")
        return city


    @listen("Faisalabad")
    def faisalabad_fact(self):
        city = "Faisalabad"

        response = completion(
            model="gemini/gemini-2.5-flash",
            api_key=os.getenv("GEMINI_API_KEY"),
            messages=[
                {
                    "role": "user",
                    "content": f"Generate 3 interesting facts about {city}."
                }
            ]
        )

        fact = response.choices[0].message.content
        print(f"\nFacts about {city}:\n")
        print(fact)


    @listen("Lahore")
    def lahore_fact(self):
        city = "Lahore"
        response = completion(
            model="gemini/gemini-2.5-flash",
            api_key=os.getenv("GEMINI_API_KEY"),
            messages=[
                {
                    "role": "user",
                    "content": f"Generate 3 interesting facts about {city}."
                }
            ]
        )

        fact = response.choices[0].message.content
        print(f"\nFacts about {city}:\n")
        print(fact)


    @listen("Karachi")
    def karachi_fact(self):
        city = "Karachi"
        response = completion(
            model="gemini/gemini-2.5-flash",
            api_key=os.getenv("GEMINI_API_KEY"),
            messages=[
                {
                    "role": "user",
                    "content": f"Generate 3 interesting facts about {city}."
                }
            ]
        )

        fact = response.choices[0].message.content
        print(f"\nFacts about {city}:\n")
        print(fact)


def start_func():
    flow = SimpleFlow()
    flow.kickoff()


def ploting():
    flow = SimpleFlow()
    flow.plot()

