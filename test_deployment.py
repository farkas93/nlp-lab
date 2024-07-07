from openai import OpenAI
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="my-token",
)

completion = client.chat.completions.create(
  model="./results_qlora",
  messages=[
    {"role": "user", "content": "Erzähl mir etwas über deine Fähigkeiten!"}
  ]
)

print(completion.choices[0].message)