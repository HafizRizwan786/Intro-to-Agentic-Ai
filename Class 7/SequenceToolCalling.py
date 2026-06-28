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
# TOOL 1
# =====================================================

def get_current_location():
    """
    Returns the user's CURRENT physical location.

    IMPORTANT:
    - Call this function ONLY IF the user DOES NOT mention any city,
      country or location in the prompt.

    Examples where this tool SHOULD be called:
    - What is the weather at my current location?
    - Where am I?
    - What's the weather here?
    - Tell me today's weather where I am.
    - Weather near me.
    - Is it raining here?

    Examples where this tool MUST NOT be called:
    - Weather in Karachi
    - Weather in Lahore
    - Weather in Islamabad
    - Temperature in London

    Returns:
        str: Current city name.
    """

    print("Tool -> get_current_location()")

    return "Lahore"


# =====================================================
# TOOL 2
# =====================================================

def get_weather(location: str):
    """
    Returns weather information for a SPECIFIC city.

    IMPORTANT:
    - If the user already mentions a city
      (for example Karachi, Lahore, London),
      call this function DIRECTLY.

    - If the user DOES NOT mention a city,
      first call get_current_location()
      and then call this function using
      the returned location.

    Examples:
    User:
        Weather in Karachi
    Action:
        get_weather("Karachi")

    User:
        Weather at my current location
    Action:
        get_current_location()
        ->
        get_weather(returned_location)

    Parameters:
        location (str): Name of the city.

    Returns:
        Weather description.
    """

    print(f"Tool -> get_weather({location})")

    location = location.lower().strip()

    weather = {
        "lahore": "40°C Sunny",
        "karachi": "34°C Cloudy",
        "islamabad": "29°C Rainy",
        "multan": "43°C Hot"
    }

    return weather.get(location, "Weather not found.")


# =====================================================
# CHAT LOOP
# =====================================================

print("=" * 50)
print("Sequential Tool Calling Example")
print("=" * 50)

while True:

    question = input("\nYou : ")

    if question.lower() == "exit":
        print('Good Bye')
        break

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=question,
            config=types.GenerateContentConfig(
                tools=[
                    get_current_location,
                    get_weather
                ],
                tool_config=types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(
                        mode="AUTO"
                    )
                )
            )
        )
        print("\nGemini:\n")
        print(response.text)
        
    except Exception as e:
        print(e)