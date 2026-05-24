from loader import load_data
from prompt import build_prompt, shuffle
from api import call_model
from parser import validate, parse_response, build_run_result

import json
import time
import pandas as pd
import os

MODELS = [
    # "openai/gpt-5.1-chat-latest",
    "deepseek/deepseek-chat",
    # "gemini/gemini-2.5-flash",
    # "anthropic/claude-haiku-4-5"
]

TOTAL_RUNS = 1000
BATCH_SIZE = 200
MAX_RETRIES = 3
OUTPUT_DIR = "outputs"

name_map = {
    "openai/gpt-5.1-chat-latest": "OpenAI",
    "deepseek/deepseek-chat": "DeepSeek",
    "gemini/gemini-2.5-flash": "Gemini",
    "anthropic/claude-haiku-4-5": "Claude"
}


def run_experiment(
    total_runs = TOTAL_RUNS,
    batch_size = BATCH_SIZE,
    models = MODELS,
    output_dir = OUTPUT_DIR
):
    df = load_data("data/term_pairs.csv")
    os.makedirs(output_dir, exist_ok=True)

    #=============================================================
    # 1000 Shuffle and then store to CSV (1000 CSV files)
    #=============================================================
    shuffled_runs = {}
    for run_id in range(1, total_runs+1):
        shuffled_df = shuffle(df, random_state=run_id)
        shuffled_runs[run_id] = shuffled_df
        shuffled_path = f"{output_dir}/shuffled_run_{run_id}.csv"
        shuffled_df.to_csv(shuffled_path, index = False)


    #===============================================================
    # Run 5 batches, each with 200 runs, total 1000 runs
    #===============================================================

    num_batches = total_runs // batch_size  # 5 batches, 200 runs each
    failed_res = [] # recording failed runs to rerun

    for batch_id in range(1, num_batches+1):  # 5 Iters
        batch_start = (batch_id - 1) * batch_size + 1
        batch_end = batch_id * batch_size

        print()
        print(f"========== BATCH {batch_id}/{num_batches}: RUN {batch_start} to {batch_end} ==========")
        
        batch_res = [] # recording the result of each batch
        for run_id in range(batch_start, batch_end+1):  # 200 Iters
            print(f"========== RUN {run_id}/{total_runs} ==========")
            shuffled_df = shuffled_runs[run_id]
            run_uuid = f"run_{run_id}"
            prompt = build_prompt(shuffled_df, run_uuid) # build prompt

            for mod in models:
                print("====================================")
                print(f"Running model: {mod}")
                print("====================================")
                parsed = None
                is_valid = False

                for attempt in range(1, MAX_RETRIES+1):
                    print(f"Attempt {attempt}/{MAX_RETRIES}")
                    try:
                        response = call_model(prompt, model = mod)
                        parsed = parse_response(response)
                    except json.JSONDecodeError as e:
                        print("JSON parse failed:", e)
                        time.sleep(2)
                        continue

                    is_valid, message = validate(parsed, shuffled_df)
                    if is_valid:
                        break

                    print("Invalid response:", message)
                    time.sleep(2)
                
                # After 3 retries:
                if not is_valid:
                    print(f"Failed to get valid response for run {run_id}, model {mod}. Skipping.")
                    failed_res.append({
                        "batch_id": batch_id,
                        "run_id": run_id,
                        "model_name": mod,
                    })
                    continue

                run_result_df = build_run_result(
                    shuffled_df = shuffled_df,
                    parsed_response = parsed,
                    run_id = run_id,
                    model_name=mod
                )

                batch_res.append(run_result_df)
        
        if batch_res:
            batch_df = pd.concat(batch_res, ignore_index = True)
            batch_path = f"{output_dir}/batch_{batch_id}_runs_{batch_start}_to_{batch_end}.csv"
            batch_df.to_csv(batch_path, index=False)
            print(f"Saved batch {batch_id} results to {batch_path}")

        else:
            print(f"No valid results collected for batch {batch_id}.")

    #==================================================================
    # failed results
    #==================================================================
    if failed_res:
        failed_df = pd.DataFrame(failed_res)
        failed_path = f"{output_dir}/failed_runs.csv"
        failed_df.to_csv(failed_path, index=False)
        print(f"Saved all failed records to {failed_path}")
    else:
        print("No failed run/model pairs.")

    #==================================================================
    # Merge all 5 batches into final_df
    #==================================================================
    all_batch_dfs = []
    for batch_id in range(1, num_batches+1):
        batch_start = (batch_id - 1) * batch_size + 1
        batch_end = batch_id * batch_size

        batch_path = f"{output_dir}/batch_{batch_id}_runs_{batch_start}_to_{batch_end}.csv"

        if os.path.exists(batch_path):  # append results of all 5 batches
            temp_df = pd.read_csv(batch_path)
            all_batch_dfs.append(temp_df)
        else:
            print(f"Warning: {batch_path} does not exist.")
    
    if all_batch_dfs:
        final_df = pd.concat(all_batch_dfs, ignore_index=True)  # concat the all_batch_dfs

        final_path = f"{output_dir}/final_flat_dataset.csv"
        final_df.to_csv(final_path, index=False)

        print("Final aggregated results:")
        print(final_df.head())
        print(f"Saved final results to {final_path}")

        return final_df
    
    else:
        print("No valid batch results were collected.")
        return None
    
if __name__ == "__main__":
    run_experiment()