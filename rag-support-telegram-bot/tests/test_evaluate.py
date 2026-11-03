import json

from rag_bot.evaluate import evaluate
from rag_bot.retrieval import Hit


class FakeKB:
    def __init__(self, results):
        self.results = results

    def search(self, question, top_k=None, threshold=None):
        return self.results[question]


def test_metrics():
    kb = FakeKB({
        "q1": [Hit("a.md", "", 0.9), Hit("b.md", "", 0.5)],   # rank 1
        "q2": [Hit("a.md", "", 0.6), Hit("b.md", "", 0.3)],   # rank 2, dropped at 0.4
    })
    cases = [{"question": "q1", "expected_file": "a.md"}, {"question": "q2", "expected_file": "b.md"}]
    low, high = evaluate(kb, cases, [0.0, 0.4], top_k=2)
    assert low == {"threshold": 0.0, "hit_rate": 1.0, "mrr": 0.75, "avg_chunks": 2.0}
    assert high == {"threshold": 0.4, "hit_rate": 0.5, "mrr": 0.5, "avg_chunks": 1.5}


def test_sample_eval_cases_point_to_existing_files(sample_dir):
    cases = json.loads((sample_dir / "eval.json").read_text(encoding="utf-8"))
    files = {p.name for p in (sample_dir / "markdown").glob("*.md")}
    assert cases and {c["expected_file"] for c in cases} <= files
