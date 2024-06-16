import pytest
import pandas as pd
from fonction import one_hot_encoder

@pytest.fixture
def sample_data():
    data = {
        'A': [1, 2, 3, 4],
        'B': ['X', 'Y', 'Z', 'X'],
        'C': [True, False, True, False]
    }
    return pd.DataFrame(data)

def test_one_hot_encoder_basic(sample_data):
    df_in = sample_data.copy()
    df_out, new_columns = one_hot_encoder(df_in)
    
    expected_columns = ['A', 'C', 'B_X', 'B_Y', 'B_Z']
    assert all(col in df_out.columns for col in expected_columns)
    assert len(new_columns) == 3  # 'B_X', 'B_Y', 'B_Z'

def test_one_hot_encoder_nan_handling(sample_data):
    df_in = sample_data.copy()
    df_in.loc[0, 'B'] = None  # Introduce NaN value
    df_out, new_columns = one_hot_encoder(df_in, nan_as_category=True)
    
    expected_columns = ['A', 'C', 'B_X', 'B_Y', 'B_Z', 'B_nan']
    assert all(col in df_out.columns for col in expected_columns)
    assert 'B_nan' in new_columns

def test_one_hot_encoder_no_nan(sample_data):
    df_in = sample_data.copy()
    df_out, new_columns = one_hot_encoder(df_in, nan_as_category=False)
    
    expected_columns = ['A', 'C', 'B_X', 'B_Y', 'B_Z']
    assert all(col in df_out.columns for col in expected_columns)
    assert 'B_nan' not in new_columns
