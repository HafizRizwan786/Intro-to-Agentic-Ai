from crewai.flow.flow import Flow,start,listen
from litellm import completion
from dotenv import load_dotenv,find_dotenv
import os
from crew_agent_task.crews.Teaching_Crew.Teaching_Crew import Teaching_Crew

_: bool=load_dotenv(find_dotenv())

class Teaching_Crew_Flow(Flow):
    @start()
    def generate_topic(self):
        response=completion(
            model=os.getenv("MODEL"),
            messages=[
                {
                    "role": "user",
                    "content": "Give me only the name of topic that is most trending in 2026"
                }
            ]
        )
        self.state['topic']=response['choices'][0]['message']['content']
    
    
    @listen(generate_topic)
    def generate_content(self):
        result = Teaching_Crew().crew().kickoff(
            inputs={"topic": self.state['topic']},
        )
        print(result)


def kickoff():
    flow = Teaching_Crew_Flow()
    flow.kickoff()
