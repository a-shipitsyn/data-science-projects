"""Turn an arXiv HTML paper into clean article text.

Paragraphs are separated by blank lines, so the splitter cuts at paragraph
boundaries first. A section title is kept on the line above its first
paragraph, so it never becomes a chunk of its own.
"""

import re

from bs4 import BeautifulSoup, Tag

# arXiv (LaTeXML) parts that are not article prose
ARXIV_DROP = [
    "ltx_bibliography", "ltx_authors", "ltx_dates", "ltx_pagination", "ltx_acknowledgements",
    "ltx_page_footer", "ltx_page_header", "ltx_role_footnote",
]
ARXIV_BLOCKS = ["ltx_title", "ltx_para", "ltx_caption", "ltx_abstract"]
SKIP_SECTION = re.compile(r"^\W*(\d+\.?\s*)?(acknowledg|funding|author contributions|competing interests)",
                          re.IGNORECASE)


def _clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([.,;:)\]])", r"\1", text)    # "( Chen et al., 2018 ) ." -> "(Chen et al., 2018)."
    return re.sub(r"([(\[])\s+", r"\1", text)


def _join(blocks: list[tuple[bool, str]]) -> str:
    """Join (is_title, text) blocks: titles end with a line break, paragraphs with a blank line."""
    out = []
    for is_title, text in blocks:
        text = _clean(text)
        if text:
            out.append(text + ("\n" if is_title else "\n\n"))
    return "".join(out).strip()


def arxiv_html_to_text(html: str | bytes) -> tuple[str, str, str]:
    """Return (title, abstract, text) of an arXiv HTML (LaTeXML) paper.

    Only the article body is read: navigation, author block, footnotes,
    acknowledgements and the reference list are dropped; formulas are replaced
    by their LaTeX source.
    """
    soup = BeautifulSoup(html, "html.parser")
    article = soup.find(class_="ltx_document") or soup.find("article")
    if article is None:
        return "", "", ""

    for name in ("script", "style", "nav", "header", "footer"):
        for node in article.find_all(name):
            node.decompose()
    for cls in ARXIV_DROP:
        for node in article.find_all(class_=cls):
            node.decompose()
    for section in article.find_all(class_=["ltx_section", "ltx_appendix", "ltx_paragraph"]):
        heading = section.find(class_="ltx_title")
        if heading and SKIP_SECTION.match(heading.get_text(" ").strip()):
            section.decompose()
    for math in article.find_all("math"):
        math.replace_with(f" {math.get('alttext', '')} ")

    title_tag = article.find(class_="ltx_title_document")
    title = _clean(title_tag.get_text(" ")) if title_tag else ""
    abstract_tag = article.find(class_="ltx_abstract")
    abstract = ""
    if abstract_tag:
        heading = abstract_tag.find(class_="ltx_title")
        if heading:
            heading.decompose()
        abstract = _clean(abstract_tag.get_text(" "))

    blocks = [(True, title), (False, f"Abstract: {abstract}" if abstract else "")]
    for node in article.find_all(class_=ARXIV_BLOCKS):
        classes = node.get("class", [])
        # skip blocks nested in a block already taken (caption inside a para, ...)
        if any(isinstance(p, Tag) and set(p.get("class", [])) & set(ARXIV_BLOCKS)
               for p in node.parents):
            continue
        if "ltx_title_document" in classes or "ltx_abstract" in classes:
            continue
        blocks.append(("ltx_title" in classes, node.get_text(" ")))
    return title, abstract, _join(blocks)
