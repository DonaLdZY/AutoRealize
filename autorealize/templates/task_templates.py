from __future__ import annotations

TASK_TEMPLATES = {
    "regression": {
        "required_sections": [
            "Overview",
            "Data Inventory",
            "Field Dictionary",
            "Task Definition",
            "Evaluation",
            "Submission Format",
            "Constraints & Risks",
        ],
        "default_metric": "RMSE",
        "default_formula": "RMSE = sqrt(mean((y_pred - y_true)^2))",
    },
    "classification": {
        "required_sections": [
            "Overview",
            "Data Inventory",
            "Field Dictionary",
            "Task Definition",
            "Evaluation",
            "Submission Format",
            "Constraints & Risks",
        ],
        "default_metric": "Accuracy",
        "default_formula": "Accuracy = (TP + TN) / (TP + TN + FP + FN)",
    },
    "optimization_or_rl": {
        "required_sections": [
            "Overview",
            "Data Inventory",
            "Field Dictionary",
            "Task Definition",
            "Evaluation",
            "Submission Format",
            "Constraints & Risks",
        ],
        "default_metric": "MatchSuccessRate",
        "default_formula": "MatchSuccessRate = matched_orders / total_orders",
    },
}


def pick_template(task_type: str) -> dict:
    t = task_type.lower()
    if "class" in t:
        return TASK_TEMPLATES["classification"]
    if "opt" in t or "match" in t or "rl" in t:
        return TASK_TEMPLATES["optimization_or_rl"]
    return TASK_TEMPLATES["regression"]
