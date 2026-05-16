from dotenv import load_dotenv
load_dotenv()
import litellm
from litellm import completion
import time

litellm.drop_params = True

def call_model(prompt, model, max_retries=3, sleep_seconds=2):
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            response = completion(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                             "You are a careful evaluator. "
                             "Return only valid JSON. "
                             "Do not include any extra text, explanations, comments, or markdown. "
                             "Do not use code fences such as ```json. "
                             "Follow the prompt instructions exactly and do not deviate from them."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ]
            )

            return response["choices"][0]["message"]["content"]

        except Exception as e:
            last_error = e
            print(f"[Retry {attempt}/{max_retries}] API call failed: {e}")
            time.sleep(sleep_seconds)

    raise RuntimeError(f"API call failed after {max_retries} retries. Last error: {last_error}")
