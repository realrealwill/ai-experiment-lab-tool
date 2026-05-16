import json
from numpy import astype
import pandas as pd


def parse_response(response_text): # JSON -> python list
    text = response_text.strip()
    if text.startswith("```json"):
        text = text.replace("```json", "", 1).strip()

    if text.endswith("```"):
        text = text[:-3].strip()

    return json.loads(text)

def build_run_result(shuffled_df, parsed_response, run_id, model_name):
    response_df = pd.DataFrame(parsed_response) # python list -> dataframe
    response_df = response_df.rename(columns={
        "id": "pair_id",
        "label": "binary_label"
    })

    merged_df = shuffled_df.merge(response_df, on="pair_id")
    merged_df = merged_df.sort_values("prompt_position").reset_index(drop=True)

    merged_df["previous_label"] = merged_df["binary_label"].shift(1).astype("Int64") # construct previous label
    merged_df["run_id"] = run_id
    merged_df["model_name"] = model_name

    return merged_df[[
        "model_name",
        "run_id",
        "term_pair",
        "prompt_position",
        "binary_label",
        "previous_label"
    ]]

