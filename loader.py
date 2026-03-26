import pandas as pd
import numpy as np

def load_data(csv_file: str) -> pd.DataFrame:
    df = pd.read_csv(csv_file)
    df["term_pair"] = df["V1"] + "," + df["V2"]
    df["term_pair_id"] = range(1, len(df) + 1)
    return df[["term_pair_id", "term_pair"]]