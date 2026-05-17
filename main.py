from loader import load_data
from prompt import build_prompt, shuffle
from api import call_model
from parser import validate, parse_response, build_run_result
import json
import time

import uuid
import pandas as pd
import os
import matplotlib.pyplot as plt

MODELS = [
    "openai/gpt-5.1-chat-latest",
    "deepseek/deepseek-chat",
    "gemini/gemini-2.5-flash",
    # "anthropic/claude-haiku-4-5"
]

RUNS = 5
MAX_RETRIES = 3

name_map = {
    "openai/gpt-5.1-chat-latest": "OpenAI",
    "deepseek/deepseek-chat": "DeepSeek",
    "gemini/gemini-2.5-flash": "Gemini",
    # "anthropic/claude-haiku-4-5": "Claude"
}


def main():
    df = load_data("data/term_pairs.csv")

    # 输出目录
    # os.makedirs("outputs", exist_ok=True)

    all_results = [] # each run, each model's result will be appended to this list

    for run_id in range(1, RUNS + 1):
        print(f"========== RUN {run_id}/{RUNS} ==========")

        shuffled_df = shuffle(df, random_state=run_id)
        # print("Shuffled DataFrame:")
        # print(shuffled_df.head())

        run_uuid = str(uuid.uuid4())
        prompt = build_prompt(shuffled_df, run_uuid)
        # shuffled_df.to_csv(f"outputs/shuffled_run_{run_id}.csv", index=False)

        for mod in MODELS:
            print("====================================")
            print(f"Running model: {mod}")
            print("====================================")

            response = call_model(prompt, model=mod)

            # print("MODEL RESPONSE:")
            # print(response)
            # print()

            parsed = None

            for attempt in range(1, MAX_RETRIES + 1):
                print(f"Attempt {attempt}/{MAX_RETRIES}")

                response = call_model(prompt, model=mod)

                try:
                    parsed = parse_response(response)
                except json.JSONDecodeError as e:
                    print("JSON parse failed:", e)
                    time.sleep(2)
                    continue

                is_valid, message = validate(parsed, shuffled_df)

                if is_valid:
                    #print("Valid response.")
                    break

                print("Invalid response:", message)
                time.sleep(2)

            # at run_id, the specific model's result
            run_result_df = build_run_result(
                shuffled_df=shuffled_df,
                parsed_response=parsed,
                run_id=run_id,
                model_name=mod
            )

            all_results.append(run_result_df)
            # print(run_result_df)


    if all_results:
        ### Flat Dataset with All Runs and Models ###
        final_df = pd.concat(all_results, ignore_index=True) # concat the list to the dataframe
        print("Final aggregated results:")
        print(final_df.head())
        # final_df.to_csv("outputs/final_flat_dataset.csv", index=False)
        # print("Saved valid results to outputs/final_flat_dataset.csv")


        ### Item Level Average Scores ###
        item_level = (
            final_df.groupby(["model_name", "term_pair"])["binary_label"]
            .mean()
            .reset_index()
            .rename(columns={"binary_label": "avg_score"})
        )
        print("Item-level average scores:")
        print(item_level.head())


        ### Transition Matrix ###
        for model_name, model_df in final_df.groupby("model_name"):
            transition_matrix = pd.crosstab(
                model_df["previous_label"],
                model_df["binary_label"],
                rownames=["Previous Label"],
                colnames=["Current Label"],
                normalize="index"
            )
            print(f"Transition matrix for {model_name}:")
            print(transition_matrix)


        ### Position Bias Curve ###
        # group by models, prompt position from 1-70, with avg score at that position calculated
        position_bias = (
            final_df.groupby(["model_name", "prompt_position"])["binary_label"]
            .mean()
            .reset_index()
            .rename(columns={"binary_label": "avg_score"})
        )
        print("Position bias curve:")
        print(position_bias)

        ### plot position bias curve ###
        # by models with prompt postion from 1 to 70
        for model_name, model_df in position_bias.groupby("model_name"):
            plt.figure(figsize=(12, 6))
            plt.plot(
                model_df["prompt_position"],
                model_df["avg_score"],
                marker="o",
                label = name_map.get(model_name, model_name)
            )

            plt.title(f"Position Bias Curve")
            plt.xlabel("Prompt Position")
            plt.ylabel("Average Score")
            plt.ylim(-0.05, 1.05)  # using the same range
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

    else:
        print("No valid results were collected.")
        
    print("Done.")


if __name__ == "__main__":
    main()