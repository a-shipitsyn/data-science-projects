import numpy as np

import schema
from importance import permutation_importance
from models import build
from monitor import drift, profile, psi


def test_psi_zero_for_same_distribution():
    assert psi([0.2, 0.3, 0.5], [0.2, 0.3, 0.5]) == 0


def test_no_shift_on_same_population(loans):
    table = drift(profile(loans.iloc[:1500]), loans.iloc[1500:])
    assert (table['shift'] != 'large').all()


def test_shift_and_missing_values_flagged(loans):
    shifted = loans.copy()
    shifted['amount'] = shifted['amount'] * 2
    shifted.loc[::3, 'existing_debt'] = np.nan
    table = drift(profile(loans), shifted).set_index('input')
    assert table.loc['amount', 'shift'] == 'large'
    assert table.loc['existing_debt', 'shift'] == 'large'
    assert 'issue_month' not in table.index          # calendar features not monitored


def test_unseen_category_counted(loans):
    new = loans.copy()
    new['merchant_category'] = 'never_seen'
    table = drift(profile(loans), new).set_index('input')
    assert table.loc['merchant_category', 'shift'] == 'large'


def test_permutation_importance(loans):
    model = build('hgb').fit(loans, loans[schema.TARGET])
    imp = permutation_importance(model, loans, repeats=2).set_index('input')
    assert set(imp.index) >= {'failed_payments_6m', 'card_expiry', 'merchant_category'}
    # the strongest simulated driver ranks above an unused window
    assert imp.loc['failed_payments_6m', 'log_loss_increase'] > imp.loc['repaid_3m', 'log_loss_increase']


def test_build_with_params(loans):
    model = build('hgb', params={'max_depth': 2, 'learning_rate': 0.05})
    assert model.estimator.named_steps['model'].max_depth == 2
    model.fit(loans, loans[schema.TARGET])
    assert 0 < model.threshold_ < 1
