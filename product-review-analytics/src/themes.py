"""Tag reviews with themes (battery, display, performance, ...) using a keyword lexicon.

A review can have several themes. Keywords are matched as whole words, case-insensitive.
"""

import argparse
import re

import pandas as pd

from data import load_reviews

THEMES = {
    "battery": ["battery", "batteries", "charge", "charging", "charger", "power"],
    "display": ["screen", "display", "resolution", "glare", "brightness", "backlight", "backlit", "ppi"],
    "performance": ["fast", "slow", "speed", "lag", "laggy", "freeze", "freezes", "crash", "crashes", "processor", "responsive"],
    "design": ["design", "light", "lightweight", "weight", "size", "compact", "sleek", "color", "colour"],
    "price": ["price", "cheap", "inexpensive", "expensive", "value", "money", "deal", "affordable", "cost", "sale"],
    "ease of use": ["easy", "simple", "intuitive", "setup", "set up", "user friendly", "user-friendly", "difficult", "complicated"],
    "sound": ["sound", "speaker", "audio", "volume", "music", "bass"],
    "voice assistant": ["alexa", "voice", "commands", "smart home", "lights", "weather"],
    "apps & content": ["app", "apps", "games", "netflix", "prime", "google play", "play store", "books", "reading", "movies", "videos"],
    "storage": ["storage", "memory", "gb", "sd card", "space"],
    "kids": ["kid", "kids", "child", "children", "son", "daughter", "grandson", "granddaughter", "parental"],
    "gift": ["gift", "christmas", "birthday", "present"],
    "connectivity": ["wifi", "wi-fi", "bluetooth", "connection", "connect", "internet", "network"],
    "support & returns": ["customer service", "support", "return", "returned", "warranty", "replacement", "refund"],
}


def compile_themes(themes: dict[str, list[str]] = THEMES) -> dict[str, re.Pattern]:
    return {
        name: re.compile(r"\b(?:" + "|".join(re.escape(k) for k in keywords) + r")\b", re.IGNORECASE)
        for name, keywords in themes.items()
    }


def theme_flags(texts: pd.Series, themes: dict[str, list[str]] = THEMES) -> pd.DataFrame:
    """One 0/1 column per theme."""
    patterns = compile_themes(themes)
    return pd.DataFrame({name: texts.str.contains(p).astype(int) for name, p in patterns.items()}, index=texts.index)


def tag_reviews(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add a `themes` column (themes found in the title and text, separated by ";") and return the 0/1 flags."""
    flags = theme_flags(df["title"] + ". " + df["text"])
    out = df.copy()
    out["themes"] = flags.apply(lambda row: ";".join(flags.columns[row.astype(bool)]), axis=1)
    return out, flags


def theme_summary(df: pd.DataFrame, flags: pd.DataFrame) -> pd.DataFrame:
    """Share of reviews per theme, mean rating with and without the theme, share of 1-3 star reviews."""
    rows = []
    for theme in flags.columns:
        has = flags[theme].astype(bool)
        rows.append({
            "theme": theme,
            "share_of_reviews": has.mean(),
            "mean_rating_with": df.loc[has, "rating"].mean(),
            "mean_rating_without": df.loc[~has, "rating"].mean(),
            "share_low_with": (df.loc[has, "rating"] <= 3).mean(),
        })
    return pd.DataFrame(rows).sort_values("share_of_reviews", ascending=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="raw review CSV")
    parser.add_argument("--output", required=True, help="CSV with one row per review and its themes")
    args = parser.parse_args()

    df = load_reviews(args.input)
    tagged, flags = tag_reviews(df)
    tagged[["product_id", "rating", "title", "text", "themes"]].to_csv(args.output, index=False)

    summary = theme_summary(df, flags)
    print(f"Reviews with at least one theme: {(flags.sum(axis=1) > 0).mean():.0%}")
    print(summary.to_string(index=False, float_format="{:.2f}".format))


if __name__ == "__main__":
    main()
