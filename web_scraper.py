import requests
from bs4 import BeautifulSoup
from bs4.element import Comment

def tag_visible(element):
    """Filter out non-visible elements like scripts/styles/comments."""
    if element.parent.name in ["style", "script", "head", "title", "meta", "[document]"]:
        return False
    if isinstance(element, Comment):
        return False
    return True

def scrape_text(url: str) -> str:

    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")

    # Get all text nodes
    texts = soup.findAll(string=True)
    visible_texts = filter(tag_visible, texts)

    # Clean and join
    cleaned = []
    for t in visible_texts:
        s = t.strip()
        if s:
            cleaned.append(s)

    return "\n".join(cleaned)
