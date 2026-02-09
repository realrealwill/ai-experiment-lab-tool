from dotenv import load_dotenv
load_dotenv()

from litellm import completion

models = [
    "openai/gpt-4o-mini",
    "deepseek/deepseek-chat"
]

for m in models:
    try:
        response = completion(
            model = m,
            messages = [{ "content": "Introduce yourself.", "role": "user" }]
        )
        print(m, "=>", response.choices[0].message.content)
    except Exception as e:
        print(m, "ERROR", e)