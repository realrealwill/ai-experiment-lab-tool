import pandas as pd
import matplotlib.pyplot as plt
import os


OUTPUT_DIR = "outputs"
name_map = {
    "openai/gpt-5.1-chat-latest": "OpenAI",
    "deepseek/deepseek-chat": "DeepSeek",
    "gemini/gemini-2.5-flash": "Gemini",
    "anthropic/claude-haiku-4-5": "Claude"
}

def run_statistics(output_dir = OUTPUT_DIR):
    final_path = f"{output_dir}/final_flat_dataset.csv"
    if not os.path.exists(final_path):
        print(f"Cannot find {final_path}. Please run experiment first.")
        return None

    final_df = pd.read_csv(final_path)
    print("Loaded final dataset:")
    print(final_df.head())

    ### Item Level Average Scores ###
    item_level = (
        final_df.groupby(["model_name", "term_pair"])["binary_label"]
        .mean()
        .reset_index()
        .rename(columns={"binary_label": "avg_score"})
    )

    item_level_path = f"{output_dir}/item_level_average_scores.csv"
    item_level.to_csv(item_level_path, index=False)
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

        safe_model_name = name_map.get(model_name, model_name).replace("/", "_")
        transition_path = f"{output_dir}/transition_matrix_{safe_model_name}.csv"
        transition_matrix.to_csv(transition_path)

    ### Position Bias Curve ###
    # group by models, prompt position from 1-70, with avg score at that position calculated
    position_bias = (
        final_df.groupby(["model_name", "prompt_position"])["binary_label"]
        .mean()
        .reset_index()
        .rename(columns={"binary_label": "avg_score"})
    )

    position_bias_path = f"{output_dir}/position_bias_curve.csv"
    position_bias.to_csv(position_bias_path, index=False)

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

        safe_model_name = name_map.get(model_name, model_name).replace("/", "_")
        figure_path = f"{output_dir}/position_bias_{safe_model_name}.png"
        plt.savefig(figure_path, dpi=300)

        plt.show()
    
    return {
        "item_level": item_level,
        "position_bias": position_bias
    }