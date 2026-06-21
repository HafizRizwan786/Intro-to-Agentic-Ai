import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# # Task 1 ------- Simple Prompt
# from google import genai

# client = genai.Client(api_key=GOOGLE_API_KEY)

# response = client.models.generate_content(
#     model="gemini-2.5-flash",
#     contents="My name is Ali"
# )

# print(response.text)





# Task 2 -------- Prompt + Image
# from google import genai
# from PIL import Image

# client = genai.Client(api_key=GOOGLE_API_KEY)

# image = Image.open("cat.png")

# response = client.models.generate_content(
#     model="gemini-2.5-flash",
#     contents=[
#         "Es pic mai jo cat ha wo konsi nasal ki ha. Hindi mai nhi answer dena.",
#         image
#     ]
# )

# print(response.text)





# Task 3 -------- Chat History
# from google import genai

# client = genai.Client(api_key=GOOGLE_API_KEY)

# chat = client.chats.create(
#     model="gemini-2.5-flash"
# )

# response = chat.send_message(
#     message="My name is Ali"
# )

# print(response.text)

# response = chat.send_message(
#     message="What is my name?"
# )

# print(response.text)
# print(chat.get_history())





# Task 4 -------- Temperature and Max Tokens
from google import genai
from google.genai import types

client = genai.Client(api_key=GOOGLE_API_KEY)

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Explain Python OOP in detail.",
    config=types.GenerateContentConfig(
        temperature=0.2,
        max_output_tokens=500
    )
)

print(response.text)