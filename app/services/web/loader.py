
import logging
import time
from collections import deque
from urllib.parse import urljoin, urlparse, urldefrag
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

USER_AGENT = "RegulaRAGBot/1.0 (+website research crawler)"
REQUEST_TIMEOUT = 20
MAX_PAGES = 40
MAX_CONTENT_LENGTH = 2_000_000

HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml",
}


class WebsiteLoaderError(RuntimeError):
    pass


def normalize_url(url: str) -> str:
    url = urldefrag(url.strip())[0]
    parsed = urlparse(url)

    if parsed.scheme not in {"http", "https"}:
        raise WebsiteLoaderError(
            "Only HTTP and HTTPS URLs are supported."
        )

    if not parsed.hostname:
        raise WebsiteLoaderError(
            "Please provide a valid website URL."
        )

    if parsed.username or parsed.password:
        raise WebsiteLoaderError(
            "URLs containing credentials are not allowed."
        )

    if parsed.port not in (None, 80, 443):
        raise WebsiteLoaderError(
            "Custom URL ports are not supported."
        )

    if parsed.hostname.lower() in {
        "localhost",
        "127.0.0.1",
        "0.0.0.0",
        "::1",
    }:
        raise WebsiteLoaderError(
            "Local URLs cannot be crawled."
        )

    return url


def extract_page(session, url: str, domain: str):
    response = session.get(
        url,
        timeout=REQUEST_TIMEOUT,
        allow_redirects=False,
        stream=True,
    )

    try:
        # Do not follow redirects outside the requested domain.
        if response.status_code in {301, 302, 303, 307, 308}:
            return None

        response.raise_for_status()

        content_type = response.headers.get(
            "Content-Type", ""
        ).lower()

        if "text/html" not in content_type:
            return None

        content_length = int(
            response.headers.get("Content-Length", "0") or 0
        )

        if content_length > MAX_CONTENT_LENGTH:
            return None

        content = bytearray()

        for block in response.iter_content(65536):
            content.extend(block)

            if len(content) > MAX_CONTENT_LENGTH:
                return None

        soup = BeautifulSoup(
            bytes(content),
            "html.parser",
        )

        for element in soup.select(
            "script, style, nav, footer, header, "
            "aside, noscript, svg, form, iframe"
        ):
            element.decompose()

        title = (
            soup.title.get_text(" ", strip=True)
            if soup.title
            else url
        )

        main = (
            soup.find("main")
            or soup.find("article")
            or soup.body
            or soup
        )

        text = main.get_text(
            separator="\n",
            strip=True,
        )

        text = "\n".join(
            line.strip()
            for line in text.splitlines()
            if line.strip()
        )

        if len(text) < 100:
            return None

        links = []

        for anchor in soup.find_all("a", href=True):
            absolute = normalize_link(
                urljoin(url, anchor["href"]),
                domain,
            )

            if absolute:
                links.append(absolute)

        return {
            "url": response.url,
            "title": title[:300],
            "text": text,
            "links": links,
        }

    finally:
        response.close()


def normalize_link(url: str, domain: str):
    try:
        url = normalize_url(url)
        parsed = urlparse(url)

        if parsed.hostname.lower() != domain:
            return None

        if parsed.path.lower().endswith(
            (
                ".pdf",
                ".png",
                ".jpg",
                ".jpeg",
                ".zip",
                ".docx",
                ".xlsx",
            )
        ):
            return None

        return url

    except (WebsiteLoaderError, ValueError):
        return None


def get_robots_parser(session, robots_url: str):
    """
    Read robots.txt before crawling.

    HTTP 2xx: parse the published crawling rules.
    HTTP 4xx: robots.txt is unavailable; proceed without rules.
    HTTP 5xx or network failure: fail closed.
    """

    robots = RobotFileParser()
    robots.set_url(robots_url)

    try:
        response = session.get(
            robots_url,
            timeout=10,
            allow_redirects=False,
        )

        try:
            status = response.status_code

            if 200 <= status < 300:
                robots.parse(response.text.splitlines())

            elif 400 <= status < 500:
                logger.warning(
                    "robots.txt returned HTTP %s. "
                    "Treating it as unavailable and proceeding "
                    "without published robots rules.",
                    status,
                )
                robots.parse([])

            else:
                raise WebsiteLoaderError(
                    f"Cannot verify crawling permissions: "
                    f"robots.txt returned HTTP {status}."
                )

        finally:
            response.close()

    except requests.RequestException as exc:
        raise WebsiteLoaderError(
            "Could not verify website crawling permissions: "
            f"{exc}"
        ) from exc

    return robots


def crawl_website(
    start_url: str,
    max_pages: int = MAX_PAGES,
):
    start_url = normalize_url(start_url)

    if not 1 <= max_pages <= MAX_PAGES:
        raise WebsiteLoaderError(
            f"Page limit must be between 1 and {MAX_PAGES}."
        )

    parsed_start = urlparse(start_url)
    domain = parsed_start.hostname.lower()

    session = requests.Session()
    session.headers.update(HEADERS)

    robots_url = (
        f"{parsed_start.scheme}://{domain}/robots.txt"
    )

    robots = get_robots_parser(
        session,
        robots_url,
    )

    queue = deque([start_url])
    visited = set()
    documents = []

    try:
        while queue and len(documents) < max_pages:

            current_url = queue.popleft()

            if current_url in visited:
                continue

            visited.add(current_url)

            if not robots.can_fetch(
                USER_AGENT,
                current_url,
            ):
                logger.info(
                    "Skipping page disallowed by robots.txt: %s",
                    current_url,
                )
                continue

            try:
                page = extract_page(
                    session,
                    current_url,
                    domain,
                )

                if not page:
                    continue

                documents.append({
                    "text": page["text"],
                    "page": None,
                    "source_url": page["url"],
                    "title": page["title"],
                    "source_type": "website",
                })

                for link in page["links"]:
                    if (
                        link not in visited
                        and link not in queue
                    ):
                        queue.append(link)

                time.sleep(0.5)

            except requests.RequestException as exc:
                logger.warning(
                    "Website page skipped (%s): %s",
                    current_url,
                    str(exc)[:200],
                )

    finally:
        session.close()

    if not documents:
        raise WebsiteLoaderError(
            "No readable website pages were extracted. "
            "Check the URL, robots.txt permissions, "
            "and website accessibility."
        )

    return documents
