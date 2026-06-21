import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
)

def ChatBot():
    print("Ask me anything! (type 'exit' to quit)")

    while True:
        question = input("Human: ")

        if question.lower() == "exit":
            print("Goodbye!")
            break

        try:
            response = llm.invoke(question)
            print("ChatBot:", response.content)

        except Exception as e:
            print("An error occurred:", str(e))


if __name__ == "__main__":
    ChatBot()