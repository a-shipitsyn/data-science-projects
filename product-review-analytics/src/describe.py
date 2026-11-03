"""Generate a short product description from the product's positive reviews with an LLM.

1. Select positive reviews (4-5 stars, not "not recommended") of one product, most helpful and
   most detailed first.
2. Ask the LLM for the 3 key features or benefits, each with a short verbatim quote as evidence;
   if a quote is not found in the reviews, the model is asked once more to replace that feature;
   quotes still not found are flagged.
3. Ask the LLM for a 2-3 sentence description that highlights only these features; numbers that
   are not in the product name or the features are flagged as possibly invented.

Any OpenAI-compatible endpoint works. The default is the free tier of Gemini (Google AI Studio);
a local model served by Ollama works too: --base-url http://localhost:11434/v1 --model llama3.2
"""

import argparse
import json
import os
import re

import pandas as pd

from data import load_reviews
from themes import theme_flags

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

FEATURES_PROMPT = """You analyse customer reviews of one product: {product}.
Below are positive reviews, one per line.

Identify the 3 most frequently praised features or benefits of the product. The 3 features
must be clearly different from each other. Use only what the reviews say. For each feature give a short verbatim quote (3-12 words)
copied exactly from one review as evidence.

Answer with JSON only, in this format:
{{"features": [{{"feature": "...", "evidence": "..."}}, ...]}}

Reviews:
{reviews}"""

RETRY_PROMPT = """These quotes were not found word for word in any review:
{missing}

Replace the features they support with features whose evidence is copied exactly, as one
continuous piece of a single review. Answer with JSON only, in the same format, all 3 features."""

DESCRIPTION_PROMPT = """Write a product description of 2-3 sentences for: {product}.
Highlight exactly these features, as customers describe them:
{features}

Use plain, factual language. Do not add any number, measurement, specification, price or claim
that is not in the product name or the features above. Answer with the description only."""


def select_reviews(df: pd.DataFrame, product: str, n: int, max_chars: int) -> tuple[str, pd.DataFrame]:
    """Positive reviews of the product whose id or name matches `product`."""
    match = (df["product_id"] == product) | df["product_name"].str.contains(product, case=False, regex=False)
    if not match.any():
        raise SystemExit(f"No product matches {product!r}")
    product_id = df.loc[match, "product_id"].value_counts().index[0]
    reviews = df[(df["product_id"] == product_id) & (df["rating"] >= 4) & (df["recommend"] != 0)]
    reviews = reviews.assign(short=reviews["text"].str.slice(0, max_chars).str.replace(r"\s+", " ", regex=True))
    reviews = reviews.sort_values(["helpful_votes", "n_words"], ascending=False).head(n)
    return reviews["product_name"].iloc[0], reviews


def theme_counts(reviews: pd.DataFrame) -> pd.Series:
    """Offline view of what the reviews talk about, to compare with the LLM's choice."""
    return theme_flags(reviews["title"] + ". " + reviews["text"]).mean().sort_values(ascending=False)


def parse_json(text: str) -> dict:
    """The JSON object in a model answer, also when it is wrapped in a code block."""
    found = re.search(r"\{.*\}", text, re.DOTALL)
    if not found:
        raise ValueError(f"No JSON in the model answer: {text[:200]}")
    return json.loads(found.group(0))


def normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", re.sub(r"\s+", " ", text.lower())).strip()


def check_evidence(features: list[dict], reviews: pd.DataFrame) -> list[dict]:
    """Mark each feature whose evidence quote (3+ words) is found in the reviews, ignoring case and punctuation."""
    corpus = normalise(" ".join(reviews["text"]))
    found = []
    for f in features:
        quote = normalise(f.get("evidence", ""))
        found.append({**f, "quote_found": len(quote.split()) >= 3 and quote in corpus})
    return found


def unsupported_numbers(description: str, product: str, features: list[dict]) -> list[str]:
    """Numbers in the description that appear neither in the product name nor in the features,
    a cheap check for invented specifications ("weighs 6.3 ounces")."""
    source = product + " " + " ".join(f"{f.get('feature', '')} {f.get('evidence', '')}" for f in features)
    allowed = set(re.findall(r"\d+(?:\.\d+)?", source))
    return [n for n in re.findall(r"\d+(?:\.\d+)?", description) if n not in allowed]


def ask(client, model: str, messages: list[dict], temperature: float) -> str:
    response = client.chat.completions.create(model=model, temperature=temperature, messages=messages)
    return response.choices[0].message.content.strip()


def extract_features(client, model: str, prompt: str, reviews: pd.DataFrame, retries: int = 1) -> list[dict]:
    """Ask for 3 features; if some quotes are not in the reviews, ask again with the failed quotes listed."""
    messages = [{"role": "user", "content": prompt}]
    for attempt in range(retries + 1):
        answer = ask(client, model, messages, temperature=0.0)
        features = check_evidence(parse_json(answer)["features"][:3], reviews)
        missing = [f["evidence"] for f in features if not f["quote_found"]]
        if not missing or attempt == retries:
            return features
        messages += [{"role": "assistant", "content": answer},
                     {"role": "user", "content": RETRY_PROMPT.format(missing="\n".join(f"- {q}" for q in missing))}]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="raw review CSV")
    parser.add_argument("--product", required=True, help="product id or part of the product name")
    parser.add_argument("--n-reviews", type=int, default=60, help="positive reviews sent to the model")
    parser.add_argument("--max-chars", type=int, default=400, help="characters kept per review")
    parser.add_argument("--base-url", default=os.getenv("LLM_BASE_URL", GEMINI_URL))
    parser.add_argument("--model", default=os.getenv("LLM_MODEL", "gemini-flash-latest"))
    parser.add_argument("--dry-run", action="store_true", help="print the first prompt, do not call the model")
    args = parser.parse_args()

    df = load_reviews(args.input)
    product, reviews = select_reviews(df, args.product, args.n_reviews, args.max_chars)
    print(f"Product: {product} ({len(reviews)} positive reviews used)")
    print("Most mentioned themes:", ", ".join(f"{t} {s:.0%}" for t, s in theme_counts(reviews).head(5).items()))

    features_prompt = FEATURES_PROMPT.format(product=product, reviews="\n".join("- " + r for r in reviews["short"]))
    if args.dry_run:
        print("\n" + features_prompt)
        return

    from openai import OpenAI

    api_key = os.getenv("LLM_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key and args.base_url == GEMINI_URL:
        raise SystemExit("Set GEMINI_API_KEY (or LLM_API_KEY for another endpoint), or use --dry-run")
    client = OpenAI(base_url=args.base_url, api_key=api_key or "local")  # local servers ignore the key

    features = extract_features(client, args.model, features_prompt, reviews)
    print("\nKey features:")
    for f in features:
        print(f"- {f['feature']}: \"{f['evidence']}\"" + ("" if f["quote_found"] else "  [quote not found in reviews]"))

    feature_list = "\n".join(f"- {f['feature']} (\"{f['evidence']}\")" for f in features)
    prompt = DESCRIPTION_PROMPT.format(product=product, features=feature_list)
    description = ask(client, args.model, [{"role": "user", "content": prompt}], temperature=0.3)
    print("\nDescription:\n" + description)
    invented = unsupported_numbers(description, product, features)
    if invented:
        print(f"[numbers not found in the product name or features: {', '.join(invented)}]")


if __name__ == "__main__":
    main()
