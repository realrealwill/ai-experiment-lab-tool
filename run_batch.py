from dotenv import load_dotenv
load_dotenv()

import csv
from datetime import datetime
from litellm import completion

MODELS = [
    "openai/gpt-4o-mini",
    "deepseek/deepseek-chat",
    "gemini/gemini-2.5-flash"
]

QUESTIONS_FILE = "questions.txt"
OUTPUT_FILE = "results.csv"

def read_questions(path: str) -> list[str]:  # read the path of the questions.txt file and return a list of questions
    questions = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:  # strip the lines and return only non-empty lines
            q = line.strip()
            if q:
                questions.append(q)
    return questions

def main():
    questions = read_questions(QUESTIONS_FILE)  # read the questions from the questions.txt file, questions is a list of strings containing the questions

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["timestamp", "model", "question_id", "question", "answer", "error"]  # the columns of the csv file, discuss what other columns we might want to add
        )
        writer.writeheader()

        for i, q in enumerate(questions):  # loop through the questions, get answers from the model, and write the results.
            for models in MODELS:
                try:
                    resp = completion(
                        model = models,
                        messages=[{"role": "user", "content": q}],
                    )
                    answer = resp.choices[0].message.content
                    err = ""
                except Exception as e:
                    answer = ""
                    err = f"{type(e).__name__}: {e}"

                writer.writerow({
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "model": models,
                    "question_id": i+1,
                    "question": q,
                    "answer": answer,
                    "error": err
                })

    print(f"Done. Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
