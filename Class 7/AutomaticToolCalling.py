import os
from dotenv import load_dotenv

from google import genai
from google.genai import types


# ==========================================================
# LOAD ENV
# ==========================================================

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY")
)


# ==========================================================
# TOOLS
# ==========================================================

def add(a: float, b: float) -> float:
    """
    Add two numbers.

    Args:
        a: First number
        b: Second number
    """
    print(f"\n✅ Tool Executed -> add({a}, {b})")
    return a + b


def subtract(a: float, b: float) -> float:
    """
    Subtract two numbers.

    Args:
        a: First number
        b: Second number
    """
    print(f"\n✅ Tool Executed -> subtract({a}, {b})")
    return a - b


def multiply(a: float, b: float) -> float:
    """
    Multiply two numbers.

    Args:
        a: First number
        b: Second number
    """
    print(f"\n✅ Tool Executed -> multiply({a}, {b})")
    return a * b


def divide(a: float, b: float):
    """
    Divide two numbers. Divide by zero is not allowed.

    Args:
        a: Dividend
        b: Divisor
    """

    print(f"\n✅ Tool Executed -> divide({a}, {b})")

    if b == 0:
        return "Division by zero is not allowed."

    return a / b


# ==========================================================
# HELPER
# ==========================================================

def generate_response(prompt):

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            # -------------------------------------
            # Register Available Tools
            # -------------------------------------
            tools=[
                add,
                subtract,
                multiply,
                divide
            ],

            # -------------------------------------
            # Tool Calling Mode
            # -------------------------------------
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    # =============================================
                    # AUTO
                    # Gemini decides whether to call a tool
                    # or answer directly.
                    #
                    # Best for AI Agents.
                    # =============================================

                    mode="AUTO"

                    # =============================================
                    # ANY
                    #
                    # Gemini MUST call one of the provided tools.
                    #
                    # Even if it could answer itself,
                    # it will still return a function call.
                    #
                    # Useful for structured extraction.
                    # =============================================

                    # mode="ANY"

                    # =============================================
                    # NONE
                    #
                    # Disable Tool Calling.
                    #
                    # Gemini behaves like a normal chatbot.
                    # =============================================

                    # mode="NONE"

                    # =============================================
                    # Allow only selected tools.
                    #
                    # Example:
                    #
                    # allowed_function_names=[
                    #     "add",
                    #     "multiply"
                    # ]
                    #
                    # Gemini CANNOT call subtract/divide.
                    # =============================================

                )
            )

            # =============================================
            # Automatic Function Calling
            #
            # Default = Enabled
            #
            # maximum_remote_calls controls how many
            # automatic tool calls the SDK can make.
            #
            # disable=True
            #     -> manual function calling
            #
            # disable=False
            #     -> automatic function calling
            # =============================================

            # automatic_function_calling=
            # types.AutomaticFunctionCallingConfig(
            #     disable=True
            # )

        )

    )

    return response


# ==========================================================
# CHAT LOOP
# ==========================================================

print("=" * 60)
print("      Automatic Tool Calling Demo")
print("=" * 60)

print("Type 'exit' to quit.\n")


while True:

    question = input("You : ")

    if question.lower() == "exit":
        print("\nGoodbye 👋")
        break

    try:

        response = generate_response(question)

        print("\nBot :", response.text)
        print()

    except Exception as e:

        print("\nError :", e)