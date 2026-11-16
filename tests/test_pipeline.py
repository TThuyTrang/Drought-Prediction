import pytest
import pandas as pd
import os
from src.drought_prediction.data_processing.ensemble_pipeline import load_and_preprocess_predictions, classify_drought
from src.drought_prediction.database.db_manager import DroughtDBManager

def test_ensemble_logic():
    df = load_and_preprocess_predictions()
    assert 'spei3_ensemble' in df.columns
    assert not df['spei3_ensemble'].isnull().all()
    # Check if ensemble is mean of others (approximately)
    sample = df.dropna(subset=['spei3_lstm', 'spei3_rf', 'spei3_xgb']).iloc[0]
    expected = (sample['spei3_lstm'] + sample['spei3_rf'] + sample['spei3_xgb']) / 3
    assert sample['spei3_ensemble'] == pytest.approx(expected)

def test_drought_classification():
    assert classify_drought(-2.5) == "Hạn cực kỳ nặng"
    assert classify_drought(-1.7) == "Hạn rất nặng"
    assert classify_drought(-1.2) == "Hạn vừa"
    assert classify_drought(0) == "Bình thường"
    assert classify_drought(1.5) == "Ẩm ướt"

def test_database_operations():
    db_path = "data/test_drought.db"
    if os.path.exists(db_path): os.remove(db_path)
    
    manager = DroughtDBManager(db_path=db_path)
    
    # Test adding alert
    manager.add_alert(2025, 1, "AnGiang", "Hạn vừa")
    
    with manager.get_connection() as conn:
        df = pd.read_sql("SELECT * FROM drought_alerts", conn)
    
    assert len(df) == 1
    assert df.iloc[0]['province'] == "AnGiang"
    
    if os.path.exists(db_path): os.remove(db_path)
