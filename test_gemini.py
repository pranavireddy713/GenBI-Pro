from google import genai

client = genai.Client(
    api_key="AQ.Ab8RN6IqrWjvysQzfnNfZmPjmysV52mzKff0UOP97nLONp5xOA"
)

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Say Hello"
)

print(response.text)