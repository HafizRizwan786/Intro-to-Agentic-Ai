import os
from dotenv import load_dotenv

from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# ==========================
# LLM
# ==========================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

# ==========================
# TOOL 1
# ==========================

@tool
def add(a: int, b: int) -> int:
    """
    Adds two integers.
    Use this tool whenever the user asks for addition.
    """
    print("\nTool Executed -> add()")
    return a + b


# ==========================
# TOOL 2
# ==========================

@tool
def multiply(a: int, b: int) -> int:
    """
    Multiplies two integers.
    Use this tool whenever the user asks for multiplication.
    """
    print("\nTool Executed -> multiply()")
    return a * b


# ==========================
# Bind Tools
# ==========================

llm_with_tools = llm.bind_tools([
    add,
    multiply
])

# ==========================
# Chat Loop
# ==========================

print("=" * 50)
print("LangChain Tool Calling")
print("=" * 50)

while True:

    question = input("\nYou : ")

    if question.lower() == "exit":
        print("Good Bye")
        break

    response = llm_with_tools.invoke(question)

    # Agar tool call hua
    if response.tool_calls:
        print("\nTool Calls:\n")
        for tool_call in response.tool_calls:
            print(tool_call)
    else:
        print("\nGemini:\n")
        print(response.content)