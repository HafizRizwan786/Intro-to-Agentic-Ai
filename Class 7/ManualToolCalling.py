import os
from dotenv import load_dotenv
from google import genai

# ==========================
# Load Environment Variables
# ==========================

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY")
)

# ==========================
# Tools
# ==========================

def add(a: int, b: int):
    return a + b


def multiply(a: int, b: int):
    return a * b


# ==========================
# Chat Loop
# ==========================

print("===== Manual Tool Calling =====")
print("Type 'exit' to quit.\n")

while True:

    question = input("You: ")

    if question.lower() == "exit":
        print("\nGoodbye!")
        break

    try:

        # -------------------------
        # Manual Tool Selection
        # -------------------------

        if "add" in question.lower():

            print("\nLLM Decision : Calculator Tool Selected")

            result = add(20, 30)

            print("Tool Executed")
            print("Tool Result :", result)

            prompt = f"""
The calculator tool returned:

{result}

Answer the user's question naturally.

User Question:
{question}
"""

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )

            print("\nGemini:", response.text)
            print()

        elif "multiply" in question.lower():

            print("\nLLM Decision : Multiply Tool Selected")

            result = multiply(20, 30)

            print("Tool Executed")
            print("Tool Result :", result)

            prompt = f"""
The multiplication tool returned:

{result}

Answer naturally.

User Question:
{question}
"""

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )

            print("\nGemini:", response.text)
            print()

        else:

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=question
            )

            print("\nGemini:", response.text)
            print()

    except Exception as e:

        print(e)