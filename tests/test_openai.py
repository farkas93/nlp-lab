from openai import OpenAI
client = OpenAI(
    base_url="http://10.42.47.11:8000/v1",
    api_key="my-token",
)

def get_response(history):
    completion = client.chat.completions.create(
        model="zskalo/gemma-1.1-2b-it-rag-sft",
        messages=history
    )
    return { "role": completion.choices[0].message.role, "content" : completion.choices[0].message.content }

history = []

init_msg = {"role": "user", "content": "Erzähl mir etwas über deine Fähigkeiten!"}
print(f"{init_msg["role"]}: {init_msg["content"]}")
history.append(init_msg)

response = get_response(history=history)
history.append(response)
print(f"{response["role"]}: {response["content"]}")

while True:
    user_input = input("user: ")
    user_msg = {"role": "user", "content": user_input}
    history.append(user_msg)
    response = get_response(history=history)
    history.append(response)
    print(f"{response["role"]}: {response["content"]}")
