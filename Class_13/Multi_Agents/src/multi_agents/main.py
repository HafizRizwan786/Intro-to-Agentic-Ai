from crewai.flow import Flow, start
from multi_agents.crews.dev_crew.dev_crew import DevCrew
from dotenv import load_dotenv,find_dotenv
load_dotenv(find_dotenv())

class DevFlow(Flow):
    
    @start()
    def run_dev_crew(self):
        try:
            print("inside the run dev crew")
            output = DevCrew().crew().kickoff(
                inputs={
                    "problem":"Write the python code of addition of two numbers"
                }
            )
            print("OUTPUT =", output)
            print("RAW =", output.raw)
            return output.raw
        except Exception as e:
            print(f"Error during kickoff: {e}")
            return str(e)


def kickoff():
    obj=DevFlow()
    result=obj.kickoff()
    print(result)