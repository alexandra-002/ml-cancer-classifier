import csv
from pathlib import Path


def save_metrics(results, filename="results/model_metrics.csv"):

    Path(filename).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(filename, "w", newline="") as file:

        writer = csv.writer(file)

        writer.writerow([
            "Metric",
            "Value"
        ])

        writer.writerow([
            "Accuracy",
            results["accuracy"]
        ])

        writer.writerow([
            "Precision",
            results["precision"]
        ])

        writer.writerow([
            "Recall",
            results["recall"]
        ])

        writer.writerow([
            "F1 Score",
            results["f1_score"]
        ])