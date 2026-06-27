#!/usr/bin/env python3
"""
HealthBridgeAI KB Ingestion Crawler
scripts/crawl_kb.py

Crawls approved disease-specific domains under restricted path/type filters
and outputs per-disease ZIP files ready for populate_kb.py ingestion.

Dependencies (install once):
    pip install requests beautifulsoup4 lxml pypdf

Usage:
    python scripts/crawl_kb.py --disease tb
    python scripts/crawl_kb.py --disease all
    python scripts/crawl_kb.py --disease tb --max-pages 30   # quick test
    python scripts/crawl_kb.py --disease tb --dry-run        # print URLs only, no download
    python scripts/crawl_kb.py --disease all --output-dir data/crawled

Output layout:
    data/crawled/
    ├── tb_knowledge_base.zip          ← feed this to populate_kb.py --disease tb
    │   ├── guideline/...txt
    │   ├── fact_sheet/...txt
    │   ├── patient_education/...txt
    │   ├── surveillance_report/...txt
    │   └── metadata.json
    ├── tb_crawl_report.json
    ├── hiv_knowledge_base.zip
    └── ...
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import re
import time
import zipfile
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse, urlunparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

try:
    from pypdf import PdfReader
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False
    print("WARNING: pypdf not installed — PDF files will be skipped. Run: pip install pypdf")

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("kb_crawler")

# ─── Global constants ─────────────────────────────────────────────────────────
DATE_CUTOFF_YEAR = 2016        # Skip content published before this year
MIN_WORD_COUNT = 80            # Skip near-empty pages / boilerplate
MAX_CONTENT_BYTES = 5_242_880  # 5 MB — skip oversized files
REQUEST_DELAY_SEC = 1.5        # Polite pause between requests to the same domain
REQUEST_TIMEOUT_SEC = 30
USER_AGENT = (
    "HealthBridgeAI-KB-Crawler/1.0 "
    "(academic/medical research bot; contact: ai-siteam@ahfid.org)"
)

# ─── URL exclusion patterns ───────────────────────────────────────────────────
# Any URL whose path contains one of these fragments (split on / - _) is skipped.
EXCLUDE_URL_FRAGMENTS: frozenset[str] = frozenset({
    # Editorial / non-clinical
    "news", "press-release", "press_release", "media-release", "media_release",
    "newsroom", "announcement",
    # Events
    "event", "events", "conference", "webinar", "seminar", "workshop",
    # Administrative
    "career", "careers", "job", "jobs", "vacancy", "vacancies",
    "about-us", "about_us", "contact", "contact-us", "contact_us",
    "donate", "donation", "fundrais", "shop", "store",
    "award", "awards",
    # Newsletter / engagement
    "newsletter", "subscribe", "subscription", "alert",
    # Site infrastructure
    "sitemap", "login", "register", "account", "sign-in", "signin",
    "privacy-policy", "privacy_policy", "terms", "cookie-policy",
    # Media
    "gallery", "photo", "video", "podcast", "webcast",
    # Social
    "twitter", "facebook", "instagram", "linkedin", "youtube",
})

# ─── Source-type classification signals ───────────────────────────────────────
# Applied to (url_path + page_title), first match wins, order matters.
SOURCE_TYPE_SIGNALS: list[tuple[str, list[str]]] = [
    ("surveillance_report", [
        "surveillance", "annual-report", "annual_report", "epidemiology",
        "prevalence", "outbreak", "incidence", "burden-of-disease",
        "global-report", "world-report", "statistics",
    ]),
    ("guideline", [
        "guideline", "guidelines", "protocol", "recommendation",
        "treatment-guideline", "treatment_guideline",
        "clinical-guidance", "consolidated-guidance",
        "standards-of-care", "management-of", "regimen", "therapy",
    ]),
    ("fact_sheet", [
        "fact-sheet", "factsheet", "fact_sheet",
        "key-facts", "overview", "briefing", "summary",
    ]),
    ("patient_education", [
        "patient", "what-is", "what_is", "basics", "understanding",
        "learn", "faq", "frequently-asked", "questions-and-answers",
    ]),
    ("research_summary", [
        "evidence", "systematic-review", "meta-analysis", "research", "findings",
    ]),
]

# ─── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class DomainConfig:
    """Crawl rules for a single domain."""
    domain: str
    seed_paths: list[str]
    include_paths: list[str]
    max_pages: int = 150
    max_depth: int = 3


@dataclass
class PageRecord:
    url: str
    domain: str
    file_type: str        # "html" | "pdf"
    source_type: str      # guideline | fact_sheet | patient_education | surveillance_report | research_summary | general
    title: str
    text: str
    content_hash: str
    word_count: int
    crawled_at: str
    published_year: Optional[int] = None

    def metadata(self) -> dict:
        """Return all fields except full text — for metadata.json."""
        d = asdict(self)
        d.pop("text")
        return d


# ─── Per-disease domain configs ───────────────────────────────────────────────

DISEASE_CONFIGS: dict[str, list[DomainConfig]] = {

    "tb": [
        DomainConfig(
            domain="who.int",
            seed_paths=[
                "/health-topics/tuberculosis/",
                "/teams/global-tuberculosis-programme/",
            ],
            include_paths=[
                "/health-topics/tuberculosis",
                "/teams/global-tuberculosis-programme",
                "/publications/",
            ],
            max_pages=80,
        ),
        DomainConfig(
            domain="cdc.gov",
            seed_paths=["/tb/"],
            include_paths=["/tb/"],
            max_pages=60,
        ),
        DomainConfig(
            domain="stoptb.org",
            seed_paths=["/resources/", "/what-we-do/"],
            include_paths=["/resources/", "/what-we-do/", "/publications/"],
            max_pages=40,
        ),
        DomainConfig(
            domain="tbfacts.org",
            seed_paths=["/"],
            include_paths=["/"],
            max_pages=60,
            max_depth=2,
        ),
        DomainConfig(
            domain="nhs.uk",
            seed_paths=["/conditions/tuberculosis-tb/"],
            include_paths=["/conditions/tuberculosis-tb/"],
            max_pages=15,
        ),
        DomainConfig(
            domain="mayoclinic.org",
            seed_paths=["/diseases-conditions/tuberculosis/"],
            include_paths=["/diseases-conditions/tuberculosis/"],
            max_pages=15,
        ),
        DomainConfig(
            domain="ntblcp.org.ng",
            seed_paths=["/resources/", "/guidelines/", "/publications/"],
            include_paths=["/resources/", "/guidelines/", "/publications/", "/downloads/"],
            max_pages=50,
        ),
        DomainConfig(
            domain="ncdc.gov.ng",
            seed_paths=["/diseases/tuberculosis/", "/guidelines/"],
            include_paths=[
                "/diseases/tuberculosis/",
                "/guidelines/tuberculosis/",
                "/reports/tb/",
            ],
            max_pages=30,
        ),
        DomainConfig(
            domain="health.gov.ng",
            seed_paths=["/tuberculosis/", "/tb/", "/publications/"],
            include_paths=["/tuberculosis/", "/tb/", "/publications/", "/guidelines/"],
            max_pages=30,
        ),
        DomainConfig(
            domain="theunion.org",
            seed_paths=["/resources/", "/what-we-do/tuberculosis/"],
            include_paths=["/resources/", "/what-we-do/tuberculosis/", "/publications/"],
            max_pages=40,
        ),
        DomainConfig(
            domain="nicd.ac.za",
            seed_paths=["/diseases-a-z-index/tuberculosis/"],
            include_paths=[
                "/diseases-a-z-index/tuberculosis/",
                "/publications/",
                "/reports/",
            ],
            max_pages=30,
        ),
        DomainConfig(
            domain="hivtb.org",
            seed_paths=["/"],
            include_paths=["/"],
            max_pages=40,
            max_depth=2,
        ),
        DomainConfig(
            domain="finddx.org",
            seed_paths=["/find-solutions/tb/"],
            include_paths=["/find-solutions/tb/", "/resources/", "/publications/"],
            max_pages=30,
        ),
        DomainConfig(
            domain="msf.org",
            seed_paths=["/en/issues/tuberculosis/"],
            include_paths=["/en/issues/tuberculosis/"],
            max_pages=25,
        ),
    ],

    "hiv": [
        DomainConfig(
            domain="who.int",
            seed_paths=[
                "/health-topics/hiv-aids/",
                "/teams/global-hiv-hepatitis-and-stis-programmes/",
            ],
            include_paths=[
                "/health-topics/hiv-aids",
                "/teams/global-hiv-hepatitis-and-stis-programmes",
                "/publications/",
            ],
            max_pages=80,
        ),
        DomainConfig(
            domain="unaids.org",
            seed_paths=["/resources/", "/topic/"],
            include_paths=["/resources/", "/topic/", "/publications/"],
            max_pages=60,
        ),
        DomainConfig(
            domain="aidsinfo.nih.gov",
            seed_paths=["/guidelines/", "/understanding-hiv-aids/"],
            include_paths=["/guidelines/", "/understanding-hiv-aids/"],
            max_pages=80,
        ),
        DomainConfig(
            domain="cdc.gov",
            seed_paths=["/hiv/"],
            include_paths=["/hiv/"],
            max_pages=60,
        ),
        DomainConfig(
            domain="aidsmap.com",
            seed_paths=["/about-hiv/"],
            include_paths=["/about-hiv/", "/page/"],
            max_pages=60,
        ),
        DomainConfig(
            domain="nhs.uk",
            seed_paths=["/conditions/hiv-and-aids/"],
            include_paths=["/conditions/hiv-and-aids/"],
            max_pages=15,
        ),
        DomainConfig(
            domain="mayoclinic.org",
            seed_paths=["/diseases-conditions/hiv-aids/"],
            include_paths=["/diseases-conditions/hiv-aids/"],
            max_pages=15,
        ),
        DomainConfig(
            domain="hiv.gov",
            seed_paths=["/hiv-basics/", "/federal-response/"],
            include_paths=["/hiv-basics/", "/federal-response/", "/understanding-hiv/"],
            max_pages=50,
        ),
        DomainConfig(
            domain="naca.gov.ng",
            seed_paths=["/resources/", "/publications/", "/hiv-information/"],
            include_paths=["/resources/", "/publications/", "/hiv-information/", "/downloads/"],
            max_pages=50,
        ),
        DomainConfig(
            domain="ncdc.gov.ng",
            seed_paths=["/diseases/hiv-aids/"],
            include_paths=[
                "/diseases/hiv-aids/",
                "/guidelines/hiv/",
                "/reports/hiv/",
            ],
            max_pages=30,
        ),
        DomainConfig(
            domain="health.gov.ng",
            seed_paths=["/hiv-aids/", "/publications/"],
            include_paths=["/hiv-aids/", "/publications/", "/guidelines/"],
            max_pages=30,
        ),
        DomainConfig(
            domain="iasociety.org",
            seed_paths=["/resources/", "/guidelines/"],
            include_paths=["/resources/", "/guidelines/", "/publications/"],
            max_pages=40,
        ),
        DomainConfig(
            domain="avert.org",
            seed_paths=["/learn-share/hiv-and-aids/"],
            include_paths=["/learn-share/hiv-and-aids/"],
            max_pages=60,
        ),
    ],

    "malaria": [
        DomainConfig(
            domain="who.int",
            seed_paths=[
                "/health-topics/malaria/",
                "/teams/global-malaria-programme/",
            ],
            include_paths=[
                "/health-topics/malaria",
                "/teams/global-malaria-programme",
                "/publications/",
            ],
            max_pages=80,
        ),
        DomainConfig(
            domain="cdc.gov",
            seed_paths=["/malaria/"],
            include_paths=["/malaria/"],
            max_pages=60,
        ),
        DomainConfig(
            domain="rollbackmalaria.org",
            seed_paths=["/resources/", "/what-we-do/"],
            include_paths=["/resources/", "/what-we-do/", "/publications/"],
            max_pages=40,
        ),
        DomainConfig(
            domain="malariaconsortium.org",
            seed_paths=["/resources/", "/what-we-do/"],
            include_paths=["/resources/", "/what-we-do/", "/publications/"],
            max_pages=40,
        ),
        DomainConfig(
            domain="nmcp.gov.ng",
            seed_paths=["/guidelines/", "/resources/", "/publications/"],
            include_paths=["/guidelines/", "/resources/", "/publications/", "/downloads/"],
            max_pages=50,
        ),
        DomainConfig(
            domain="ncdc.gov.ng",
            seed_paths=["/diseases/malaria/"],
            include_paths=[
                "/diseases/malaria/",
                "/guidelines/malaria/",
                "/reports/malaria/",
            ],
            max_pages=30,
        ),
        DomainConfig(
            domain="health.gov.ng",
            seed_paths=["/malaria/", "/publications/"],
            include_paths=["/malaria/", "/publications/", "/guidelines/"],
            max_pages=30,
        ),
        DomainConfig(
            domain="nicd.ac.za",
            seed_paths=["/diseases-a-z-index/malaria/"],
            include_paths=[
                "/diseases-a-z-index/malaria/",
                "/publications/",
                "/reports/",
            ],
            max_pages=30,
        ),
        DomainConfig(
            domain="afro.who.int",
            seed_paths=["/health-topics/malaria/", "/publications/"],
            include_paths=["/health-topics/malaria/", "/publications/"],
            max_pages=40,
        ),
    ],
}

# ─── Utility functions ────────────────────────────────────────────────────────

def canonical_url(url: str) -> str:
    """Normalise: lowercase scheme+host, strip fragment, keep path as-is."""
    p = urlparse(url)
    return urlunparse((p.scheme.lower(), p.netloc.lower(), p.path, "", "", ""))


def url_path_allowed(url: str, include_paths: list[str]) -> bool:
    path = urlparse(url).path
    return any(path.startswith(inc) for inc in include_paths)


def url_excluded(url: str) -> bool:
    """Return True if the URL path contains any excluded fragment."""
    path = urlparse(url).path.lower()
    tokens = set(re.split(r"[/\-_.]", path))
    return bool(tokens & EXCLUDE_URL_FRAGMENTS)


def is_crawlable_extension(url: str) -> bool:
    """Only follow links to HTML pages or PDFs."""
    path = urlparse(url).path.lower()
    return path.endswith(("", "/", ".html", ".htm", ".php", ".asp", ".aspx", ".pdf"))


def classify_source_type(url: str, title: str) -> str:
    text = (urlparse(url).path + " " + title).lower()
    for stype, signals in SOURCE_TYPE_SIGNALS:
        if any(s in text for s in signals):
            return stype
    return "general"


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:20]


def extract_year(text: str) -> Optional[int]:
    """Find the most recent year (2010–now) mentioned in the first 3 000 chars."""
    candidates = re.findall(r"\b(20[1-3]\d)\b", text[:3000])
    years = [int(y) for y in candidates if int(y) <= datetime.now().year]
    return max(years) if years else None


def extract_year_from_soup(soup: BeautifulSoup) -> Optional[int]:
    """Prefer meta tags, fall back to body text."""
    for tag in soup.find_all("meta"):
        val = tag.get("content", "") + tag.get("name", "") + tag.get("property", "")
        m = re.search(r"\b(20[1-3]\d)\b", val)
        if m:
            return int(m.group(1))
    for tag in soup.find_all("time"):
        m = re.search(r"\b(20[1-3]\d)\b", tag.get("datetime", "") + tag.get_text())
        if m:
            return int(m.group(1))
    return extract_year(soup.get_text(" ", strip=True))


def extract_main_text(soup: BeautifulSoup) -> str:
    """Strip chrome, extract readable content."""
    for tag in soup(["script", "style", "nav", "header", "footer",
                     "aside", "form", "button", "noscript", "iframe",
                     "svg", "figure", "picture"]):
        tag.decompose()

    # Class/id patterns that indicate navigation or promotional chrome
    chrome_re = re.compile(
        r"(nav|menu|sidebar|banner|cookie|popup|modal|breadcrumb|social|"
        r"share|footer|header|ad[-_]|advertisement|promo|widget|skip-link|"
        r"site-header|page-header|toc|table-of-contents)",
        re.I,
    )
    for tag in soup.find_all(True):
        if not isinstance(getattr(tag, "attrs", None), dict):
            continue
        cls = " ".join(tag.get("class", []))
        eid = tag.get("id", "")
        if chrome_re.search(cls) or chrome_re.search(eid):
            tag.decompose()

    # Try semantic containers first
    for selector in [
        "main", "article", '[role="main"]',
        "#main-content", "#content", ".main-content",
        ".article-body", ".entry-content", ".page-content",
    ]:
        node = soup.select_one(selector)
        if node:
            return node.get_text(" ", strip=True)

    return soup.body.get_text(" ", strip=True) if soup.body else ""


def clean_text(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf_text(content: bytes) -> str:
    if not HAS_PYPDF:
        return ""
    try:
        reader = PdfReader(io.BytesIO(content))
        pages = []
        for page in reader.pages:
            try:
                pages.append(page.extract_text() or "")
            except Exception:
                continue
        return clean_text("\n".join(pages))
    except Exception as exc:
        log.debug("PDF parse error: %s", exc)
        return ""


# ─── Robots.txt cache ─────────────────────────────────────────────────────────

class RobotsCache:
    def __init__(self) -> None:
        self._cache: dict[str, RobotFileParser] = {}

    def allowed(self, url: str) -> bool:
        parsed = urlparse(url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        if base not in self._cache:
            rp = RobotFileParser(f"{base}/robots.txt")
            try:
                rp.read()
            except Exception:
                pass  # If robots.txt unreachable, assume allowed
            self._cache[base] = rp
        return self._cache[base].can_fetch(USER_AGENT, url)


# ─── Crawler ──────────────────────────────────────────────────────────────────

class KBCrawler:
    def __init__(
        self,
        disease: str,
        output_dir: Path,
        dry_run: bool = False,
        max_pages_override: Optional[int] = None,
    ) -> None:
        self.disease = disease
        self.output_dir = output_dir
        self.dry_run = dry_run
        self.max_pages_override = max_pages_override

        self.configs = DISEASE_CONFIGS[disease]
        self.robots = RobotsCache()
        self.records: list[PageRecord] = []
        self._seen_urls: set[str] = set()
        self._seen_hashes: set[str] = set()
        self._domain_last_req: dict[str, float] = defaultdict(float)

        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/pdf",
            "Accept-Language": "en-US,en;q=0.9",
        })

    # ── HTTP ──────────────────────────────────────────────────────────────────

    def _throttle(self, domain: str) -> None:
        elapsed = time.monotonic() - self._domain_last_req[domain]
        if elapsed < REQUEST_DELAY_SEC:
            time.sleep(REQUEST_DELAY_SEC - elapsed)
        self._domain_last_req[domain] = time.monotonic()

    def _fetch(self, url: str) -> Optional[requests.Response]:
        domain = urlparse(url).netloc
        self._throttle(domain)
        try:
            r = self._session.get(url, timeout=REQUEST_TIMEOUT_SEC, stream=True)
            r.raise_for_status()
            length = int(r.headers.get("Content-Length", 0))
            if length > MAX_CONTENT_BYTES:
                log.debug("Skip (too large %d bytes): %s", length, url)
                r.close()
                return None
            # Read body now (stream=True defers it; enforce size cap)
            content = b""
            for chunk in r.iter_content(chunk_size=65536):
                content += chunk
                if len(content) > MAX_CONTENT_BYTES:
                    log.debug("Skip (streaming too large): %s", url)
                    r.close()
                    return None
            r._content = content  # type: ignore[attr-defined]
            return r
        except requests.RequestException as exc:
            log.debug("Fetch failed: %s — %s", url, exc)
            return None

    # ── Domain crawl ──────────────────────────────────────────────────────────

    def _crawl_domain(self, cfg: DomainConfig) -> int:
        max_pages = self.max_pages_override or cfg.max_pages
        queue: deque[tuple[str, int]] = deque()

        for path in cfg.seed_paths:
            seed = canonical_url(f"https://{cfg.domain}{path}")
            if seed not in self._seen_urls:
                self._seen_urls.add(seed)
                queue.append((seed, 0))

        crawled = 0
        log.info("[%s] Crawling %-30s  max=%d  depth=%d",
                 self.disease, cfg.domain, max_pages, cfg.max_depth)

        while queue and crawled < max_pages:
            url, depth = queue.popleft()

            if not self.robots.allowed(url):
                log.debug("robots.txt blocks: %s", url)
                continue

            if self.dry_run:
                log.info("  [dry-run] %s", url)
                crawled += 1
                continue

            r = self._fetch(url)
            if r is None:
                continue

            ctype = r.headers.get("Content-Type", "")
            is_pdf = "pdf" in ctype or url.lower().endswith(".pdf")

            if is_pdf:
                record = self._process_pdf(url, r.content)
            elif "html" in ctype:
                record = self._process_html(url, r.content, depth, queue, cfg)
            else:
                continue

            if record:
                self.records.append(record)
                crawled += 1
                log.info("  %3d/%-3d  %-18s  %5d words  %s",
                         crawled, max_pages, record.source_type,
                         record.word_count, url[:72])

        return crawled

    # ── HTML processing ───────────────────────────────────────────────────────

    def _process_html(
        self,
        url: str,
        content: bytes,
        depth: int,
        queue: deque,
        cfg: DomainConfig,
    ) -> Optional[PageRecord]:
        try:
            soup = BeautifulSoup(content, "lxml")
        except Exception:
            soup = BeautifulSoup(content, "html.parser")

        # Enqueue child links before modifying soup
        if depth < cfg.max_depth:
            for a in soup.find_all("a", href=True):
                href = str(a["href"]).strip()
                if href.startswith(("mailto:", "tel:", "javascript:", "#")):
                    continue
                child = canonical_url(urljoin(url, href))
                parsed = urlparse(child)
                if (
                    parsed.scheme in ("http", "https")
                    and parsed.netloc.endswith(cfg.domain)
                    and child not in self._seen_urls
                    and url_path_allowed(child, cfg.include_paths)
                    and not url_excluded(child)
                    and is_crawlable_extension(child)
                ):
                    self._seen_urls.add(child)
                    queue.append((child, depth + 1))

        title = soup.title.string.strip() if soup.title and soup.title.string else ""
        raw_text = extract_main_text(soup)
        text = clean_text(raw_text)

        if len(text.split()) < MIN_WORD_COUNT:
            return None

        year = extract_year_from_soup(soup)
        if year is not None and year < DATE_CUTOFF_YEAR:
            log.debug("Outdated (%d), skip: %s", year, url)
            return None

        h = hash_text(text)
        if h in self._seen_hashes:
            log.debug("Duplicate content, skip: %s", url)
            return None
        self._seen_hashes.add(h)

        return PageRecord(
            url=url,
            domain=urlparse(url).netloc,
            file_type="html",
            source_type=classify_source_type(url, title),
            title=title,
            text=text,
            content_hash=h,
            word_count=len(text.split()),
            crawled_at=datetime.utcnow().isoformat(),
            published_year=year,
        )

    # ── PDF processing ────────────────────────────────────────────────────────

    def _process_pdf(self, url: str, content: bytes) -> Optional[PageRecord]:
        text = extract_pdf_text(content)
        if len(text.split()) < MIN_WORD_COUNT:
            return None

        year = extract_year(text)
        if year is not None and year < DATE_CUTOFF_YEAR:
            log.debug("Outdated PDF (%d), skip: %s", year, url)
            return None

        h = hash_text(text)
        if h in self._seen_hashes:
            return None
        self._seen_hashes.add(h)

        title = Path(urlparse(url).path).stem.replace("-", " ").replace("_", " ")

        return PageRecord(
            url=url,
            domain=urlparse(url).netloc,
            file_type="pdf",
            source_type=classify_source_type(url, title),
            title=title,
            text=text,
            content_hash=h,
            word_count=len(text.split()),
            crawled_at=datetime.utcnow().isoformat(),
            published_year=year,
        )

    # ── Run + save ────────────────────────────────────────────────────────────

    def run(self) -> None:
        for cfg in self.configs:
            self._crawl_domain(cfg)

        if self.dry_run:
            log.info("[%s] dry-run complete — %d URLs queued",
                     self.disease, len(self._seen_urls))
            return

        self._save()

    def _save(self) -> None:
        if not self.records:
            log.warning("[%s] No documents collected — nothing to save.", self.disease)
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)
        zip_path = self.output_dir / f"{self.disease}_knowledge_base.zip"

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for rec in self.records:
                # Folder = source_type so populate_kb.py can filter at ingest time
                safe = re.sub(r"[^\w\-]", "_", urlparse(rec.url).path.strip("/"))[:80]
                fname = f"{rec.source_type}/{safe}.txt"
                zf.writestr(fname, rec.text)

            meta_list = [r.metadata() for r in self.records]
            zf.writestr("metadata.json", json.dumps(meta_list, indent=2, ensure_ascii=False))

        # Crawl report (summary statistics)
        by_type: dict[str, int] = defaultdict(int)
        by_domain: dict[str, int] = defaultdict(int)
        for r in self.records:
            by_type[r.source_type] += 1
            by_domain[r.domain] += 1

        report = {
            "disease": self.disease,
            "crawled_at": datetime.utcnow().isoformat(),
            "total_documents": len(self.records),
            "total_words": sum(r.word_count for r in self.records),
            "by_source_type": dict(sorted(by_type.items())),
            "by_domain": dict(sorted(by_domain.items(), key=lambda x: -x[1])),
            "surveillance_note": (
                "Documents in source_type=surveillance_report contain statistics/prevalence data. "
                "They are included in the ZIP but the ingestion pipeline should filter them out "
                "for clinical (non-statistics) query intents."
            ),
        }

        report_path = self.output_dir / f"{self.disease}_crawl_report.json"
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

        log.info(
            "[%s] Saved %d documents → %s  (%.2f MB)",
            self.disease, len(self.records), zip_path,
            zip_path.stat().st_size / 1_048_576,
        )
        log.info("[%s] Crawl report → %s", self.disease, report_path)
        log.info("[%s] By type: %s", self.disease,
                 "  ".join(f"{k}={v}" for k, v in sorted(by_type.items())))
        log.info("[%s] By domain: %s", self.disease,
                 "  ".join(f"{d.split('.')[0]}={n}" for d, n in
                           sorted(by_domain.items(), key=lambda x: -x[1])[:5]))


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Crawl approved medical domains to build HealthBridgeAI knowledge bases.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--disease",
        choices=[*DISEASE_CONFIGS.keys(), "all"],
        required=True,
        help="Disease KB to crawl. Use 'all' for every configured disease.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print URLs that would be crawled without making any HTTP requests.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        metavar="N",
        help="Override max pages per domain (useful for quick smoke tests, e.g. --max-pages 5).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/crawled"),
        help="Base directory for crawled output ZIPs and reports.",
    )
    args = parser.parse_args()

    if not HAS_PYPDF:
        log.warning("pypdf not installed — PDF files will be skipped. pip install pypdf")

    diseases = list(DISEASE_CONFIGS.keys()) if args.disease == "all" else [args.disease]

    for disease in diseases:
        log.info("=" * 60)
        log.info("  KB crawl: %s", disease.upper())
        log.info("=" * 60)
        crawler = KBCrawler(
            disease=disease,
            output_dir=args.output_dir,
            dry_run=args.dry_run,
            max_pages_override=args.max_pages,
        )
        crawler.run()

    log.info("All done.")


if __name__ == "__main__":
    main()
