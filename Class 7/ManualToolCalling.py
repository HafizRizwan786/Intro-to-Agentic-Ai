import os
from dotenv import load_dotenv

from google import genai
from google.genai import types


# =====================================================
# LOAD ENV
# =====================================================

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY")
)

# =====================================================
# PYTHON FUNCTIONS
# =====================================================

def add(a: float, b: float):

    return a + b


def subtract(a: float, b: float):

    return a - b


def multiply(a: float, b: float):

    return a * b


def divide(a: float, b: float):

    if b == 0:
        return "Division by zero is not allowed."

    return a / b


# =====================================================
# TOOL DECLARATIONS
# =====================================================

calculator_tool = types.Tool(

    function_declarations=[

        types.FunctionDeclaration(

            name="add",

            description="Add two numbers.",

            parameters_json_schema={
                "type": "object",
                "properties": {
                    "a": {
                        "type": "number"
                    },
                    "b": {
                        "type": "number"
                    }
                },
                "required": [
                    "a",
                    "b"
                ]
            }
        ),

        types.FunctionDeclaration(

            name="subtract",

            description="Subtract two numbers.",

            parameters_json_schema={
                "type": "object",
                "properties": {
                    "a": {
                        "type": "number"
                    },
                    "b": {
                        "type": "number"
                    }
                },
                "required": [
                    "a",
                    "b"
                ]
            }
        ),

        types.FunctionDeclaration(

            name="multiply",

            description="Multiply two numbers.",

            parameters_json_schema={
                "type": "object",
                "properties": {
                    "a": {
                        "type": "number"
                    },
                    "b": {
                        "type": "number"
                    }
                },
                "required": [
                    "a",
                    "b"
                ]
            }
        ),

        types.FunctionDeclaration(

            name="divide",

            description="Divide two numbers.",

            parameters_json_schema={
                "type": "object",
                "properties": {
                    "a": {
                        "type": "number"
                    },
                    "b": {
                        "type": "number"
                    }
                },
                "required": [
                    "a",
                    "b"
                ]
            }
        )

    ]

)

# =====================================================
# FUNCTION MAP
# =====================================================

tool_map = {

    "add": add,
    "subtract": subtract,
    "multiply": multiply,
    "divide": divide

}

# =====================================================
# CHAT LOOP
# =====================================================

print("=" * 60)
print("Manual Tool Calling")
print("=" * 60)

while True:

    question = input("\nYou : ")

    if question.lower() == "exit":
        print("Good Bye")
        break
        

    # ----------------------------------------------
    # USER MESSAGE
    # ----------------------------------------------

    user_content = types.Content(
        role="user",
        parts=[
            types.Part.from_text(
                text=question
            )
        ]
    )

    # ----------------------------------------------
    # GEMINI
    # ----------------------------------------------

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[user_content],
        config=types.GenerateContentConfig(
            tools=[calculator_tool],
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode="ANY"
                )
            )
        )
    )

    # ----------------------------------------------
    # MODEL FUNCTION CALL
    # ----------------------------------------------

    function_call = response.function_calls[0]

    print("\n========== FUNCTION CALL ==========")
    print("Tool :", function_call.name)
    print("Arguments :", function_call.args)

    # ----------------------------------------------
    # EXECUTE PYTHON FUNCTION
    # ----------------------------------------------

    result = tool_map[function_call.name](
        **function_call.args
    )

    print("\n========== TOOL RESULT ==========")
    print(result)

    # ----------------------------------------------
    # CREATE FUNCTION RESPONSE
    # ----------------------------------------------

    function_response = types.Part.from_function_response(
        name=function_call.name,
        response={
            "result": result
        }
    )

    # ----------------------------------------------
    # SEND RESULT BACK TO GEMINI
    # ----------------------------------------------

    final_response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            user_content,
            response.candidates[0].content,
            types.Content(
                role="tool",
                parts=[
                    function_response
                ]
            )
        ],
        config=types.GenerateContentConfig(
            tools=[calculator_tool]
        )
    )

    print("\n========== FINAL ANSWER ==========")
    print(final_response.text)