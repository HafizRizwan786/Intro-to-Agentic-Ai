from crewai import Agent,Crew,Task
from crewai.project import CrewBase,agent,crew,task

@CrewBase
class Teaching_Crew():
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"
    
    @agent
    def sir_zia(self):
        return Agent(
            config=self.agents_config["sir_zia"]
        )
    
    @task
    def describe_topic(self):
        return Task(
            config=self.tasks_config["describe_topic"],
        )
    
    @crew
    def crew(self):
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            verbose=True
        )