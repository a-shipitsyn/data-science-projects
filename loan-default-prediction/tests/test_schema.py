import pytest

import schema


def test_simulated_data_is_valid(loans):
    schema.validate(loans, outcomes=True)
    assert 0.04 < loans[schema.TARGET].mean() < 0.15


def test_missing_column(loans):
    with pytest.raises(ValueError, match='missing columns'):
        schema.validate(loans.drop(columns=['amount']))


def test_all_problems_reported(loans):
    bad = loans.head(10).copy()
    bad.loc[0, 'amount'] = -5
    bad.loc[1, 'failed_payments_3m'] = bad.loc[1, 'failed_payments_6m'] + 1
    bad.loc[2, 'card_exp_month'] = 13
    with pytest.raises(ValueError) as e:
        schema.validate(bad)
    msg = str(e.value)
    assert 'amount' in msg and 'failed_payments_3m' in msg and 'card_exp_month' in msg


def test_optional_columns_may_be_missing(loans):
    df = loans.head(5).copy()
    df['existing_debt'] = float('nan')
    schema.validate(df)


def test_target_from_balance_after_grace(loans):
    assert (loans[schema.TARGET] == (loans['outstanding_after_grace'] > 0)).all()
