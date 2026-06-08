from __future__ import annotations

import os
import io as _io
from typing import Generator

import numpy as np


def load_csv(
    filepath: str,
    delimiter: str = ",",
    dtype: type = float,
    missing_values: str = "",
    filling_values: float = np.nan,
    skip_header: int = 0,
) -> np.ndarray:
    """Load a CSV file into a 2-D NumPy array.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.
    delimiter : str
        Column separator. Default ','.
    dtype : data-type
        Output dtype. Default float.
    missing_values : str
        Token treated as missing. Default '' (blank cell).
    filling_values : float
        Value used to fill missing entries. Default np.nan.
    skip_header : int
        Number of header rows to skip. Default 0.

    Returns
    -------
    np.ndarray
        2-D array of shape (n_rows, n_cols).

    Raises
    ------
    FileNotFoundError
        If filepath does not exist.
    ValueError
        If the file is empty.

    Time complexity: O(n) where n = total cells.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No such file or directory: '{filepath}'")
    if os.path.getsize(filepath) == 0:
        raise ValueError(f"File is empty: '{filepath}'")
    data = np.genfromtxt(
        filepath,
        delimiter=delimiter,
        dtype=dtype,
        missing_values=missing_values,
        filling_values=filling_values,
        skip_header=skip_header,
        autostrip=True,
    )
    if data.ndim == 0:
        raise ValueError(f"File could not be parsed: '{filepath}'")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data


def load_csv_chunked(
    filepath: str,
    chunksize: int,
    delimiter: str = ",",
    dtype: type = float,
    missing_values: str = "",
    filling_values: float = np.nan,
    skip_header: int = 0,
) -> Generator[np.ndarray, None, None]:
    """Stream a large CSV file in fixed-size row chunks.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.
    chunksize : int
        Max rows per yielded chunk. Must be >= 1.
    delimiter : str
        Column separator. Default ','.
    dtype : data-type
        Output dtype. Default float.
    missing_values : str
        Token treated as missing. Default ''.
    filling_values : float
        Value used to fill missing entries. Default np.nan.
    skip_header : int
        Number of header rows to skip before chunking.

    Yields
    ------
    np.ndarray
        2-D array of shape (<=chunksize, n_cols).

    Raises
    ------
    FileNotFoundError
        If filepath does not exist.
    ValueError
        If chunksize < 1.

    Notes
    -----
    Never reads the whole file at once.
    Time complexity: O(chunksize * n_cols) per chunk.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No such file or directory: '{filepath}'")
    if chunksize < 1:
        raise ValueError(f"chunksize must be >= 1, got {chunksize}.")

    def _parse(lines):
        arr = np.genfromtxt(
            _io.StringIO("".join(lines)),
            delimiter=delimiter,
            dtype=dtype,
            missing_values=missing_values,
            filling_values=filling_values,
            autostrip=True,
        )
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr

    with open(filepath, "r", encoding="utf-8") as fh:
        for _ in range(skip_header):
            next(fh, None)
        buffer: list[str] = []
        for line in fh:
            buffer.append(line)
            if len(buffer) == chunksize:
                yield _parse(buffer)
                buffer = []
        if buffer:
            yield _parse(buffer)


def save_csv(
    array: np.ndarray,
    filepath: str,
    delimiter: str = ",",
    header: str | None = None,
) -> None:
    """Write a NumPy array to a CSV file.

    Parameters
    ----------
    array : np.ndarray
        1-D or 2-D array to save.
    filepath : str
        Destination path.
    delimiter : str
        Column separator. Default ','.
    header : str or None
        Header row string. Default None.

    Raises
    ------
    ValueError
        If array is not 1-D or 2-D.

    Time complexity: O(n) where n = total cells.
    """
    if array.ndim not in (1, 2):
        raise ValueError(
            f"array must be 1-D or 2-D, got {array.ndim}-D with shape {array.shape}."
        )
    np.savetxt(
        filepath,
        array,
        delimiter=delimiter,
        header=header if header is not None else "",
        comments="",
    )
