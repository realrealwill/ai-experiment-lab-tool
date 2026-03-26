from loader import load_data
from prompt import build_prompt
from prompt import shuffle
from api import call_model
import uuid

MODEL_NAME = [
    "openai/gpt-5.1-chat-latest",
    "deepseek/deepseek-chat",
    "gemini/gemini-2.5-flash",
    "anthropic/claude-haiku-4-5"
]

def main():
    ##### LOAD DATA FROM CSV #####
    df = load_data("data/term_pairs.csv")
    shuffled_df = shuffle(df).head(5)
    run_uuid = str(uuid.uuid4())

    prompt = build_prompt(shuffled_df, run_uuid)

    for mod in MODEL_NAME:
        print("====================================")
        print(f"Running model: {mod}")
        print("====================================")
        response = call_model(prompt, model=mod)
        print("MODEL RESPONSE:")
        print(response)
        print()

if __name__ == "__main__":
    main()