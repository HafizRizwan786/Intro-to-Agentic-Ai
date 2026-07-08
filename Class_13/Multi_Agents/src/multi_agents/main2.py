from crewai.flow import Flow, start
from multi_agents.crews.dev_crew2.dev_crew2 import DevCrew2
from dotenv import load_dotenv,find_dotenv
load_dotenv(find_dotenv())

class DevFlow2(Flow):
    
    @start()
    def run_dev_crew(self):
        try:
            print("inside the run dev crew")
            output = DevCrew2().crew().kickoff(
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


def kickoff2():
    obj=DevFlow2()
    result=obj.kickoff()
    print(result)