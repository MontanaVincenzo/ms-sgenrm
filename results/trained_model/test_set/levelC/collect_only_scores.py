import json
from pathlib import Path
import pandas as pd

from typing import Any, Dict, List, Optional
import pandas as pd


def json_to_dataframe(
    data: List[Dict[str, Any]],
    normalize: bool = False,
    record_path: Optional[str | List[str]] = None,
    meta: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Convert a list of JSON objects (dictionaries) into a pandas DataFrame.

    Parameters:
        data: List of dictionaries representing JSON objects.
        normalize: If True, flattens nested JSON structures using pd.json_normalize.
        record_path: Path in each object to the list of records (used if normalize=True).
        meta: Fields to use as metadata for each record in resulting table (used if normalize=True).

    Returns:
        pd.DataFrame: Resulting pandas DataFrame, or an empty DataFrame if data is empty.
    """
    if not data:
        return pd.DataFrame()

    if normalize:
        return pd.json_normalize(data, record_path=record_path, meta=meta)

    return pd.DataFrame.from_records(data)

if __name__ == '__main__':

    input_file = Path("trained_model.jsonl")

    records = [json.loads(l) for l in input_file.read_text(encoding="utf-8").splitlines() if l.strip()]

    dataset = []

    for sample in records:
        for p_key in ["pipeline1", "pipeline2"]:
            d = {
                "annotation_id": sample["annotation_id"],
                "pipeline_key": p_key,
                "asr_perturbation": sample[p_key]["perturbation"]["asr"],
                "llm_perturbation": sample[p_key]["perturbation"]["llm"],
                "tts_perturbation": sample[p_key]["perturbation"]["tts"],
                "llm_evaluator_score": sample[p_key]["gt_evaluation"]["overall_score"],
                "reward_model_score": sample[p_key]["model_evaluation"]["overall_score"],
                "mae_wrt_llm": abs(sample[p_key]["gt_evaluation"]["overall_score"] - sample[p_key]["model_evaluation"]["overall_score"])
            }
            dataset.append(d)
    df = json_to_dataframe(dataset)

    df.to_csv("./scores.csv")
    