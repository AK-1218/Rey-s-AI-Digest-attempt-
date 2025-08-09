import time
import re
import requests
from typing import Optional, List
from bs4 import BeautifulSoup
import trafilatura
import google.generativeai as genai

# -------- Paste your Gemini API key here --------
GEMINI_API_KEY = "AIzaSyB8zh7WIxH_0IvA-hujjPfAJRnDcEqEmGk"  # e.g., "AIzaSy...."

# -------- Config --------
LISTING_URL = "https://www.artificialintelligence-news.com/artificial-intelligence-news/"
NUM_ARTICLES = 2
PRINT_SPACED = True
PRINT_FULL_TEXT = False
TRIM_ARTICLE_CHARS = 60000
MODEL_NAME = "gemini-1.5-pro"

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

DEFAULT_LINK_SELECTORS: List[str] = [
    ".elementor-loop-container .elementor-widget-theme-post-title h1 a",
    ".elementor-loop-container h1 a",
    "article h1 a",
]

CUSTOM_LINK_SELECTORS: List[str] = []  # optional overrides

# -------- Helpers --------
def init_gemini(api_key: str, model_name: str) -> genai.GenerativeModel:
    if not api_key or not api_key.startswith("AIza"):
        raise RuntimeError("Gemini API key is missing or invalid.")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name)

def get_listing_html(url: str) -> str:
    r = requests.get(url, headers=REQUEST_HEADERS, timeout=20)
    r.raise_for_status()
    return r.text

def first_n_links(listing_html: str, n: int, custom_selectors: Optional[List[str]] = None) -> List[str]:
    soup = BeautifulSoup(listing_html, "lxml")
    selectors = (custom_selectors or []) + DEFAULT_LINK_SELECTORS
    links: List[str] = []
    for sel in selectors:
        for a in soup.select(sel):
            href = (a.get("href") or "").strip()
            if href and href.startswith("http"):
                links.append(href)
        if links:
            break
    seen, uniq = set(), []
    for href in links:
        if href not in seen:
            seen.add(href)
            uniq.append(href)
    return uniq[:n]

def extract_text(url: str, spaced: bool = True) -> str:
    r = requests.get(url, headers=REQUEST_HEADERS, timeout=30)
    r.raise_for_status()
    txt = trafilatura.extract(r.text, include_comments=False, include_tables=False)
    if txt and txt.strip():
        return normalize_paragraphs(txt) if spaced else txt.strip()
    soup = BeautifulSoup(r.text, "lxml")
    main = soup.select_one(".td-post-content, .entry-content, article") or soup
    paras = [p.get_text(" ", strip=True) for p in main.find_all(["p", "li"])]
    if paras:
        return "\n\n".join(paras) if spaced else "\n".join(paras)
    return main.get_text(" ", strip=True)

def normalize_paragraphs(text: str) -> str:
    paras = [p.strip() for p in text.splitlines() if p.strip()]
    return "\n\n".join(paras)

def summarize_with_gemini(model: genai.GenerativeModel, text: str) -> str:
    if not text or len(text) < 200:
        return text.strip()
    trimmed = text[:TRIM_ARTICLE_CHARS]
    prompt = (
        "Summarize the following article in EXACTLY 3 sentences.\n"
        "- No fluff or hype. Keep it factual and easy to understand.\n"
        "- Include the most important numbers if present.\n"
        "- No opinions, no predictions.\n\n"
        "Article:\n"
        f"{trimmed}\n\n"
        "Now write the 3-sentence summary:"
    )
    for attempt in range(3):
        try:
            resp = model.generate_content(prompt)
            out = (resp.text or "").strip()
            sents = re.split(r'(?<=[.!?])\s+', out)
            sents = [s.strip() for s in sents if s.strip()]
            return " ".join(sents[:3])
        except Exception as e:
            if attempt == 2:
                raise
            time.sleep(1.5 * (attempt + 1))

# -------- Main --------
def main():
    model = init_gemini(GEMINI_API_KEY, MODEL_NAME)
    listing_html = get_listing_html(LISTING_URL)
    links = first_n_links(listing_html, NUM_ARTICLES, CUSTOM_LINK_SELECTORS)
    if not links:
        raise SystemExit("No links found.")
    for i, url in enumerate(links, 1):
        print(f"\n=== Article {i}/{len(links)} ===\n{url}\n")
        try:
            article_text = extract_text(url, spaced=PRINT_SPACED)
            if PRINT_FULL_TEXT:
                preview = article_text[:3000] + ("\n...\n" if len(article_text) > 3000 else "\n")
                print(preview)
            summary = summarize_with_gemini(model, article_text)
            print("— Summary (3 sentences) —\n")
            print(summary + "\n")
        except Exception as e:
            print(f"[warn] Failed {url}: {e}")
        time.sleep(1.0)

if __name__ == "__main__":
    main()
