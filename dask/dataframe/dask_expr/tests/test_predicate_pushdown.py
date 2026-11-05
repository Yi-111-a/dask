from __future__ import annotations

import pytest

from dask.dataframe.dask_expr import from_pandas
from dask.dataframe.dask_expr._expr import rewrite_filters
from dask.dataframe.dask_expr.tests._util import _backend_library, assert_eq

pd = _backend_library()


@pytest.fixture
def pdf():
    pdf = pd.DataFrame({"x": range(100), "a": 1, "b": 2})
    pdf["y"] = pdf.x // 7  # Not unique; duplicates span different partitions
    yield pdf


@pytest.fixture
def df(pdf):
    yield from_pandas(pdf, npartitions=10)


def test_rewrite_filters(df):
    predicate = (df.x == 1) | (df.x == 1)
    expected = df.x == 1
    assert rewrite_filters(predicate.expr)._name == expected._name

    predicate = (df.x == 1) | ((df.x == 1) & (df.y == 2))
    expected = df.x == 1
    assert rewrite_filters(predicate.expr)._name == expected._name

    predicate = ((df.x == 1) & (df.y == 3)) | ((df.x == 1) & (df.y == 2))
    expected = (df.x == 1) & ((df.y == 3) | (df.y == 2))
    assert rewrite_filters(predicate.expr)._name == expected._name

    predicate = ((df.x == 1) & (df.y == 3) & (df.a == 1)) | (
        (df.x == 1) & (df.y == 2) & (df.a == 1)
    )
    expected = ((df.x == 1) & (df.a == 1)) & ((df.y == 3) | (df.y == 2))
    assert rewrite_filters(predicate.expr)._name == expected._name

    predicate = (df.x == 1) | (df.y == 1)
    assert rewrite_filters(predicate.expr)._name == predicate._name

    predicate = df.x == 1
    assert rewrite_filters(predicate.expr)._name == predicate._name

    predicate = (df.x.isin([1, 2, 3]) & (df.y == 3)) | (
        df.x.isin([1, 2, 3]) & (df.y == 2)
    )
    expected = df.x.isin([1, 2, 3]) & ((df.y == 3) | (df.y == 2))
    assert rewrite_filters(predicate.expr)._name == expected._name


def test_rewrite_filters_query(df, pdf):
    result = df[((df.x == 1) & (df.y > 1)) | ((df.x == 1) & (df.y > 2))]
    result = result[["x"]]
    expected = pdf[((pdf.x == 1) & (pdf.y > 1)) | ((pdf.x == 1) & (pdf.y > 2))]
    expected = expected[["x"]]
    assert_eq(result, expected)


def test_rewrite_filters_bool_identity(df):
    # https://github.com/dask/dask/issues/12664
    # `False | (x == y)` builds an Or holding the raw bool; the identity operand
    # must fold out so the rewriter never dereferences a bool's `_name`.
    predicate = False | (df.x == 1)
    expected = df.x == 1
    assert rewrite_filters(predicate.expr)._name == expected._name

    # `True & (x == y)` never reaches the OR rewriter; it must still compute
    # with pandas element-wise semantics (asserted in
    # test_filter_with_bool_scalar_operand).


@pytest.mark.parametrize("scalar, kind", [(False, "or"), (True, "or"), (False, "and"), (True, "and")])
def test_filter_with_bool_scalar_operand(df, pdf, scalar, kind):
    # https://github.com/dask/dask/issues/12664
    # Both sides must match pandas element-wise bool semantics, including the
    # identities (`False | x`, `True & x`) and the absorbing operands
    # (`True | x`, `False & x`).
    result = df[scalar | (df.x == 1)] if kind == "or" else df[scalar & (df.x == 1)]
    expected = (
        pdf[scalar | (pdf.x == 1)] if kind == "or" else pdf[scalar & (pdf.x == 1)]
    )
    assert_eq(result, expected)


def test_filter_bool_identity_with_common_or_component(df, pdf):
    # A folded identity inside an or-component shared by EVERY branch must not
    # derail the common-component replacement.
    result = df[((df.x >= 1) & (False | (df.y == 3))) | ((df.x < 6) & (df.y == 2))]
    expected = pdf[((pdf.x >= 1) & (False | (pdf.y == 3))) | ((pdf.x < 6) & (pdf.y == 2))]
    assert_eq(result, expected)
