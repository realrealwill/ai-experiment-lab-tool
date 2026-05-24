from prompt import build_prompt
from api import call_model
from parser import validate, parse_response, build_run_result

import json
import time
import pandas as pd
import os


MAX_RETRIES = 3
OUTPUT_DIR = "outputs"


def rerun(
    failed_path=f"{OUTPUT_DIR}/failed_runs.csv",
    output_dir=OUTPUT_DIR
):

    if not os.path.exists(failed_path):
        print(f"No failed runs to rerun.")
        return None

    failed_df = pd.read_csv(failed_path)

    if failed_df.empty:
        print("No failed runs to rerun.")
        return None

    rerun_results = []
    still_failed = []

    for _, row in failed_df.iterrows():
        batch_id = int(row["batch_id"])
        run_id = int(row["run_id"])
        mod = row["model_name"]

        print()
        print("====================================")
        print(f"Rerunning failed case:")
        print(f"Batch: {batch_id}")
        print(f"Run: {run_id}")
        print(f"Model: {mod}")
        print("====================================")

        # Use the saved shuffled file
        shuffled_path = f"{output_dir}/shuffled_run_{run_id}.csv"

        if not os.path.exists(shuffled_path):
            print(f"Cannot find this file. Skipping.")

            still_failed.append({
                "batch_id": batch_id,
                "run_id": run_id,
                "model_name": mod
            })
            continue

        shuffled_df = pd.read_csv(shuffled_path)

        parsed = None
        is_valid = False

        for attempt in range(1, MAX_RETRIES + 1):
            print(f"Attempt {attempt}/{MAX_RETRIES}")

            run_uuid = f"rerun_failed_{run_id}_attempt_{attempt}"
            prompt = build_prompt(shuffled_df, run_uuid)

            try:
                response = call_model(prompt, model=mod)
                parsed = parse_response(response)

            except json.JSONDecodeError as e:
                print("JSON parse failed:", e)
                time.sleep(2)
                continue

            except Exception as e:
                print("Model call or parsing failed:", e)
                time.sleep(2)
                continue

            is_valid, message = validate(parsed, shuffled_df)

            if is_valid:
                print("Valid response.")
                break

            print("Invalid response:", message)

            # ===== DEBUG: find missing and extra pair ids =====
            if isinstance(parsed, list):
                expected_ids = set(shuffled_df["pair_id"].astype(str))

                returned_ids = set(
                    str(item.get("id")) for item in parsed if isinstance(item, dict)
                )

                missing_ids = expected_ids - returned_ids
                extra_ids = returned_ids - expected_ids

                print("Missing ids:", missing_ids)
                print("Extra ids:", extra_ids)

            time.sleep(2)

        if not is_valid:
            print(f"Still failed: run {run_id}, model {mod}")

            still_failed.append({
                "batch_id": batch_id,
                "run_id": run_id,
                "model_name": mod,
            })

            continue

        run_result_df = build_run_result(
            shuffled_df=shuffled_df,
            parsed_response=parsed,
            run_id=run_id,
            model_name=mod
        )

        rerun_results.append(run_result_df)

    if rerun_results:
        rerun_df = pd.concat(rerun_results, ignore_index=True)

        rerun_path = f"{output_dir}/fixed_results.csv"
        rerun_df.to_csv(rerun_path, index=False)

        print(f"Saved rerun fixed results to {rerun_path}")

        merge(output_dir)  # merge the rerun results to the final flat dataset

    else:
        print("No rerun results were fixed.")

    if still_failed:
        still_failed_df = pd.DataFrame(still_failed)

        still_failed_path = f"{output_dir}/still_failed_runs.csv"
        still_failed_df.to_csv(still_failed_path, index=False)

        print(f"Saved still failed records to {still_failed_path}")
    else:
        print("No still-failed records.")

    return rerun_results

def merge(output_dir = OUTPUT_DIR):
    final_path = f"{output_dir}/final_flat_dataset.csv"
    rerun_path = f"{output_dir}/fixed_results.csv"

    if not os.path.exists(final_path):
        print(f"Cannot find {final_path}.")
        return None

    if not os.path.exists(rerun_path):
        print(f"Cannot find {rerun_path}.")
        return None
    
    final_df = pd.read_csv(final_path)
    rerun_df = pd.read_csv(rerun_path)

    merge_df = pd.concat([final_df, rerun_df], ignore_index = True)
    merge_df = merge_df.drop_duplicates(
        subset=["run_id", "model_name", "term_pair"],
        keep="last"
    )

    merge_df = merge_df.sort_values(
        by = ["run_id", "model_name", "prompt_position"]
    ).reset_index(drop = True)

    merge_path = f"{output_dir}/final_flat_dataset_merged.csv"
    merge_df.to_csv(merge_path, index = False)

    print("merge complete: ", merge_df.shape)
    print(f"merge saved to {merge_path}")

    return merge_df

if __name__ == "__main__":
    rerun()