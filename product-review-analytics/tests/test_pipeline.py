from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from data import clean_name, coarse_category
from describe import check_evidence, extract_features, parse_json, select_reviews, unsupported_numbers
from eda import rating_factors
from rating_model import NUMERIC, build_model, cross_validate_model, features, with_tuning
from themes import tag_reviews, theme_flags
from topics import STOP_WORDS, key_phrases, nmf_topics

# data.py

def test_loader_drops_incomplete_rows_and_personal_fields(reviews):
    assert len(reviews) == 298
    assert reviews["rating"].between(1, 5).all()
    assert not {"reviews.username", "reviews.sourceURLs", "username"} & set(reviews.columns)


def test_loader_fills_missing_product_names(reviews):
    assert reviews["product_name"].notna().all()
    assert reviews.groupby("product_id")["product_name"].nunique().eq(1).all()


@pytest.mark.parametrize("text, expected", [
    ("Kids Tablet, 7 inch, with Kid-Proof Case", "tablet"),
    ("Charger and Power Adapter for Fire Tablets and E-readers", "accessory"),
    ("Reader Oasis E-reader with Leather Charging Cover", "e-reader"),
    ("Echo (White)", "smart speaker"),
    ("Streaming Media Players,Electronics", "streaming"),
    ("Rice Dishes,Ready Meals", "other"),
])
def test_coarse_category(text, expected):
    assert coarse_category(text) == expected


def test_clean_name_removes_duplicate_and_commas():
    assert clean_name("Echo (White),,,\r\nEcho (White),,,") == "Echo (White)"


# themes.py

def test_themes_match_whole_words_only():
    flags = theme_flags(pd.Series(["Battery is great", "recharged overnight", "Kids love it", "the app crashes"]))
    assert flags.loc[0, "battery"] == 1
    assert flags.loc[1, "battery"] == 0  # "recharged" is not "charge"
    assert flags.loc[2, "kids"] == 1
    assert flags.loc[3, "performance"] == 1 and flags.loc[3, "apps & content"] == 1


def test_tag_reviews_lists_all_themes(reviews):
    tagged, flags = tag_reviews(reviews)
    assert len(tagged) == len(reviews)
    assert (tagged["themes"].str.split(";").str.len().where(tagged["themes"] != "", 0) == flags.sum(axis=1)).all()


# eda.py

def test_rating_factors_ranks_all_features(reviews):
    factors, r2 = rating_factors(reviews)
    assert "recommend" not in set(factors["factor"])
    assert factors["importance"].is_monotonic_decreasing
    assert -1 < r2 <= 1


# rating_model.py

def test_features_have_text_and_structured_columns(reviews):
    X = features(reviews)
    assert set(NUMERIC + ["category", "review", "recommend"]) <= set(X.columns)
    assert set(X["recommend"].dropna()) <= {"yes", "no"}


def test_cross_validation_returns_out_of_fold_predictions(reviews):
    X, y = features(reviews), reviews["rating"].to_numpy()
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    model = build_model(numeric=NUMERIC, categorical=["category", "recommend"])
    pred, folds, chosen = cross_validate_model(model, X, y, cv, c_grid=[0.5, 2.0])
    assert pred.shape == y.shape and set(pred) <= set(y)
    assert len(folds) == 3 and len(chosen) == 3 and set(chosen) <= {0.5, 2.0}
    assert folds["macro_f1"].mean() > 0.5  # the synthetic texts are easy to separate


def test_single_c_switches_tuning_off():
    model = with_tuning(build_model(), [3.0])
    assert not isinstance(model, GridSearchCV)
    assert model.get_params()["logisticregression__C"] == 3.0
    assert isinstance(with_tuning(build_model(), [1.0, 2.0]), GridSearchCV)


# topics.py

def test_key_phrases_do_not_start_or_end_with_stop_words():
    texts = pd.Series(["battery life is easy to set up"] * 30 + ["the screen is bright"] * 30)
    phrases = key_phrases(texts, top=5)["phrase"]
    stop = set(STOP_WORDS)
    assert all(p.split()[0] not in stop and p.split()[-1] not in stop for p in phrases)
    assert "battery life" in set(phrases)


def test_nmf_topics_cover_reviews(reviews):
    topics, dominant = nmf_topics(reviews["text"], n_topics=3, n_words=5)
    assert len(topics) == 3
    assert topics["share_of_reviews"].sum() == pytest.approx(np.mean(dominant >= 0))


# describe.py

def test_select_reviews_takes_positive_reviews_of_one_product(reviews):
    name, selected = select_reviews(reviews, "Speaker", n=10, max_chars=20)
    assert name == "Home Speaker with voice assistant"
    assert selected["product_id"].eq("P2").all() and (selected["rating"] >= 4).all()
    assert selected["short"].str.len().max() <= 20


def test_select_reviews_unknown_product():
    with pytest.raises(SystemExit):
        select_reviews(pd.DataFrame({"product_id": ["P1"], "product_name": ["A"]}), "zzz", 5, 100)


def test_parse_json_inside_code_block():
    answer = 'Sure:\n```json\n{"features": [{"feature": "Battery", "evidence": "lasts for weeks"}]}\n```'
    assert parse_json(answer)["features"][0]["feature"] == "Battery"
    with pytest.raises(ValueError):
        parse_json("no json here")


def test_check_evidence_flags_invented_and_too_short_quotes():
    reviews = pd.DataFrame({"text": ["The screen is GREAT, easy on the eyes!"]})
    checked = check_evidence([
        {"feature": "screen", "evidence": "screen is great, easy on the eyes"},
        {"feature": "battery", "evidence": "battery lasts for weeks"},
        {"feature": "short", "evidence": "screen"},
    ], reviews)
    assert [c["quote_found"] for c in checked] == [True, False, False]


def test_unsupported_numbers_flags_invented_specs():
    features = [{"feature": "Long battery", "evidence": "lasts 3 weeks on a charge"}]
    description = "The 6 inch reader lasts 3 weeks and weighs only 6.3 ounces."
    assert unsupported_numbers(description, "Reader, 6 inch display", features) == ["6.3"]


class FakeClient:
    """Stands in for the OpenAI client: returns the given answers in order and records the requests."""

    def __init__(self, answers):
        self.answers, self.requests = list(answers), []
        self.chat = self.completions = self

    def create(self, **request):
        self.requests.append(request)
        message = SimpleNamespace(content=self.answers.pop(0))
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_extract_features_retries_once_when_a_quote_is_invented():
    reviews = pd.DataFrame({"text": ["The screen is great and easy on the eyes. Battery lasts for weeks."]})
    bad = '{"features": [{"feature": "screen", "evidence": "screen is great"}, {"feature": "games", "evidence": "millions of trivia games"}]}'
    good = '{"features": [{"feature": "screen", "evidence": "screen is great"}, {"feature": "battery", "evidence": "battery lasts for weeks"}]}'
    client = FakeClient([bad, good])
    features = extract_features(client, "model", "prompt", reviews)
    assert [f["quote_found"] for f in features] == [True, True]
    assert len(client.requests) == 2
    assert "millions of trivia games" in client.requests[1]["messages"][-1]["content"]
