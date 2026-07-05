from multi_agents.crews.dev_crew.dev_crew import DevCrew
from dotenv import load_dotenv
import traceback

load_dotenv()

try:
    print("Starting crew kickoff...")
    crew = DevCrew().crew()
    result = crew.kickoff(inputs={"problem": "Write the python code of addition of two numbers"})
    print("Crew completed successfully.")
    print("Result:", result.raw)
except Exception as e:
    print("An error occurred:")
    traceback.print_exc()
