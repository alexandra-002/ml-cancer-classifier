import csv
from pathlib import Path


def save_metrics(results, model_name):

    filename = Path("results/model_metrics.csv")

    file_exists = filename.exists()

    with open(filename, "a", newline="") as file:

        writer = csv.writer(file)

        if not file_exists:
            writer.writerow(
                [
                    "Model",
                    "Accuracy",
                    "Precision",
                    "Recall",
                    "F1_Score"
                ]
            )

        writer.writerow(
            [
                model_name,
                round(results["accuracy"], 3),
                round(results["precision"], 3),
                round(results["recall"], 3),
                round(results["f1_score"], 3)
            ]
        )