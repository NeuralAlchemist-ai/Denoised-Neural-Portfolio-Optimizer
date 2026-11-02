import pandas as pd
import pytest
import yaml

from portfolio.data import download


@pytest.fixture
def dates() -> pd.DatetimeIndex:
    return pd.bdate_range("2024-01-01", periods=3)


def test_download_retries_an_incomplete_ticker_and_persists_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path, dates: pd.DatetimeIndex
):
    initial = pd.DataFrame({"A": [100.0, 101.0, 102.0], "B": [200.0, None, 202.0]}, index=dates)
    retry = pd.DataFrame({"B": [200.0, 201.0, 202.0]}, index=dates)
    calls: list[list[str]] = []

    def fake_fetch(tickers: list[str], **_: object) -> pd.DataFrame:
        calls.append(list(tickers))
        return initial if tickers == ["A", "B"] else retry

    monkeypatch.setattr(download, "_fetch_adj_close", fake_fetch)
    monkeypatch.setattr(download.time, "sleep", lambda _: None)
    output_path = tmp_path / "prices.parquet"

    result = download.download_prices(["A", "B"], "2024-01-01", "2024-01-10", output_path)

    assert calls == [["A", "B"], ["B"]]
    assert result.loc[dates[1], "B"] == 201.0
    pd.testing.assert_frame_equal(pd.read_parquet(output_path), result, check_freq=False)
    metadata = yaml.safe_load(output_path.with_suffix(".metadata.yaml").read_text())
    assert metadata["field"] == "Adj Close"
    assert metadata["tickers"] == ["A", "B"]


def test_download_rejects_unresolved_incomplete_history(
    monkeypatch: pytest.MonkeyPatch, tmp_path, dates: pd.DatetimeIndex
):
    incomplete = pd.DataFrame({"A": [100.0, None, 102.0]}, index=dates)
    monkeypatch.setattr(download, "_fetch_adj_close", lambda *_, **__: incomplete)

    with pytest.raises(RuntimeError, match="complete valid"):
        download.download_prices(
            ["A"], "2024-01-01", "2024-01-10", tmp_path / "prices.parquet", max_retries=0
        )


@pytest.mark.parametrize(
    ("tickers", "start", "end", "filename", "message"),
    [
        ([], "2024-01-01", "2024-01-10", "prices.parquet", "non-empty"),
        (["A", "A"], "2024-01-01", "2024-01-10", "prices.parquet", "duplicates"),
        (["A"], "2024-01-10", "2024-01-01", "prices.parquet", "earlier"),
        (["A"], "2024-01-01", "2024-01-10", "prices.csv", ".parquet"),
    ],
)
def test_download_rejects_invalid_inputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    tickers: list[str],
    start: str,
    end: str,
    filename: str,
    message: str,
):
    monkeypatch.setattr(download, "_fetch_adj_close", lambda *_, **__: pytest.fail("not called"))

    with pytest.raises(ValueError, match=message):
        download.download_prices(tickers, start, end, tmp_path / filename)


def test_validate_data_config_rejects_asset_count_mismatch():
    config = {
        "date_range": {"start": "2024-01-01", "end": "2024-01-10"},
        "ticker_list": ["A", "B"],
        "number_of_assets": 1,
    }

    with pytest.raises(ValueError, match="must match"):
        download.validate_data_config(config)