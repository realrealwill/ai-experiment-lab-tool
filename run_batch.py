from dotenv import load_dotenv
load_dotenv()

import csv
from datetime import datetime
from litellm import completion

# MODEL = "openai/gpt-4o-mini"
MODEL = "deepseek/deepseek-chat"

QUESTIONS_FILE = "questions.txt"
OUTPUT_FILE = "results.csv"

def read_questions(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def main():
    questions = read_questions(QUESTIONS_FILE)

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["timestamp", "model", "question_id", "question", "answer", "error"]
        )
        writer.writeheader()

        for i, q in enumerate(questions):
            try:
                resp = completion(
                    model=MODEL,
                    messages=[{"role": "user", "content": q}],
                )
                answer = resp.choices[0].message.content
                err = ""
            except Exception as e:
                answer = ""
                err = f"{type(e).__name__}: {e}"

            writer.writerow({
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "model": MODEL,
                "question_id": i,
                "question": q,
                "answer": answer,
                "error": err
            })

    print(f"Done. Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
