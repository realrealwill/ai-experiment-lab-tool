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

def validate(parsed, shuffled_df):
    # must be a list
    if not isinstance(parsed, list):
        return False, "Parsed response is not a list."
    
    # items not missing
    expect_ids = shuffled_df["pair_id"].tolist()
    if len(parsed) != len(expect_ids):
        return False, f"Expected {len(expect_ids)} items, got {len(parsed)}."
    
    # each item in the list
    output_ids = []
    for i, item in enumerate(parsed):
        # each item be a dict
        if not isinstance(item, dict):
            return False, f"Item {i} is not a correct object"
        # contain id and label
        if "id" not in item or "label" not in item:
            return False, f"Item {i} has invalid label."
    
    #duplication
    if len(set(output_ids)) != len(output_ids):
        return False, "Duplication"
    
    return True, "Valid response."
    

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

