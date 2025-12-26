import pandas as pd
import pytest
from pathlib import Path

from churn_compass.io import FileIO


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "int_col": [1, 2, 3], 
            "float_col": [0.1, 0.2, 0.3], 
            "str_col": ["a", "b", "c"], 
            "bool_col": [True, False, True], 
        }
    )

@pytest.fixture
def file_io():
    return FileIO()

def test_write_and_read_parquet(tmp_path: Path, sample_df, file_io):
    """Parquet round-trip should preserve data and schema"""
    path = tmp_path / "test.parquet"

    file_io.write_parquet(sample_df, path)
    assert path.exists()

    df_read = file_io.read_parquet(path)

    pd.testing.assert_frame_equal(df_read, sample_df)

def test_write_and_read_csv(tmp_path: Path, sample_df, file_io):
    """
    CSV round-trip should preserve values.
    """
    path = tmp_path / "test.csv"

    file_io.write_csv(sample_df, path)
    assert path.exists()

    df_read = file_io.read_csv(path)

    pd.testing.assert_frame_equal(df_read, sample_df)


def test_read_nonexistent_file_raises(tmp_path: Path, file_io):
    """
    Reading a missing file should raise a clear error.
    """
    path = tmp_path / "missing.parquet"

    with pytest.raises(Exception):
        file_io.read_parquet(path)


def test_overwrite_existing_file(tmp_path: Path, sample_df, file_io):
    """
    Writing to an existing path should overwrite cleanly.
    """
    path = tmp_path / "overwrite.parquet"

    file_io.write_parquet(sample_df, path)
    file_io.write_parquet(sample_df, path)

    df_read = file_io.read_parquet(path)
    pd.testing.assert_frame_equal(df_read, sample_df)


def test_invalid_input_type_raises(file_io):
    """
    Writing non-DataFrame input should fail loudly.
    """
    with pytest.raises(Exception):
        file_io.write_parquet("not_a_df", "dummy.parquet")