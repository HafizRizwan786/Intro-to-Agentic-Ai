from crewai import Agent,Crew,Task
from crewai.project import CrewBase,agent,crew,task
from crewai import Process
from crewai.llm import LLM
import os

llm1=LLM(
    model='gemini/gemini-2.5-flash',
    api_key=os.getenv("GEMINI_API_KEY")
)

llm2=LLM(
    model="gemini/gemini-2.5-flash",
    api_key=os.getenv("GEMINI_API_KEY")
)


@CrewBase
class DevCrew():
    agents_config="config/agents.yaml"
    tasks_config="config/tasks.yaml"
    
    @agent
    def junior_python_developer(self):
        return Agent(
            config=self.agents_config['junior_python_developer'],
            llm=llm1
        )

    @agent
    def senior_python_developer(self):
        return Agent(
            config=self.agents_config['senior_python_developer'],
            llm=llm2
        )

    @task
    def code_generation(self):
        return Task(
            config=self.tasks_config['code_generation']
        )
    
    @task
    def code_review(self):
        return Task(
            config=self.tasks_config['code_review']
        )
    
    
    @crew
    def crew(self):
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )