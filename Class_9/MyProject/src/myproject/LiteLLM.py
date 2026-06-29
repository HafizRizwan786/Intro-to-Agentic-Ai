from dotenv import load_dotenv
import os
from litellm import completion

load_dotenv()

def gemini_model():
    response = completion(
        model="gemini/gemini-2.5-flash",
        api_key=os.getenv("GEMINI_API_KEY"),
        messages=[
            {
                "role": "user",
                "content": "Hello! Tell me about Agentic AI."
            }
        ]
    )

    print(response.choices[0].message.content)