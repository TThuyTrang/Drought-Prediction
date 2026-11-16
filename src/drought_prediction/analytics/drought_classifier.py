import pandas as pd
import numpy as np

def classify_spei3(val: float) -> tuple[str, str]:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return ("Không có dữ liệu", "#9E9E9E")
    if val > -0.5:
        return ("Bình thường", "#2E7D32")
    elif -1.0 <= val <= -0.5:
        return ("Hạn nhẹ", "#F9A825")
    elif -1.5 <= val < -1.0:
        return ("Hạn vừa", "#EF6C00")
    else:
        return ("Hạn nặng đến cực đoan", "#B71C1C")

def classify_series(series: pd.Series) -> pd.DataFrame:
    results = series.apply(classify_spei3)
    return pd.DataFrame(results.tolist(), index=series.index, columns=["severity", "color"])
