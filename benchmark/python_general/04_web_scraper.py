"""
Web scraper using requests and BeautifulSoup.
Handles session management, retries, robots.txt, pagination, throttling.
"""
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Optional, Generator
from urllib.parse import urljoin, urlparse
from pathlib import Path
import json
import time
import logging
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RobotsTxtChecker:
    """Check if a URL is allowed by robots.txt"""

    def __init__(self, user_agent: str = "WiseScraper/1.0"):
        self.user_agent = user_agent
        self.cache: Dict[str, bool] = {}

    def can_fetch(self, url: str) -> bool:
        """Check if URL can be fetched according to robots.txt"""
        domain = urlparse(url).netloc
        robots_url = f"http://{domain}/robots.txt"

        if robots_url in self.cache:
            return self.cache[robots_url]

        try:
            resp = requests.get(robots_url, timeout=5)
            if resp.status_code == 200:
                rules = self._parse_robots(resp.text, domain)
                allowed = self._check_rules(url, rules)
                self.cache[robots_url] = allowed
                return allowed
        except Exception as e:
            logger.warning(f"Failed to fetch robots.txt for {domain}: {e}")
            self.cache[robots_url] = True
            return True

        self.cache[robots_url] = True
        return True

    def _parse_robots(self, content: str, domain: str) -> Dict[str, List[str]]:
        """Parse robots.txt content"""
        rules = {"disallow": [], "allow": []}
        in_user_agent = False

        for line in content.split("\n"):
            line = line.split("#")[0].strip()
            if not line:
                continue

            if line.lower().startswith("user-agent:"):
                agent = line.split(":", 1)[1].strip()
                in_user_agent = agent == "*" or agent == self.user_agent

            elif in_user_agent:
                if line.lower().startswith("disallow:"):
                    path = line.split(":", 1)[1].strip()
                    if path:
                        rules["disallow"].append(path)
                elif line.lower().startswith("allow:"):
                    path = line.split(":", 1)[1].strip()
                    if path:
                        rules["allow"].append(path)

        return rules

    def _check_rules(self, url: str, rules: Dict[str, List[str]]) -> bool:
        """Check if URL matches disallow rules"""
        path = urlparse(url).path
        for disallow in rules["disallow"]:
            if path.startswith(disallow):
                return False
        return True


class WebScraper:
    """Web scraper with retry logic, session management, throttling"""

    def __init__(
        self,
        base_url: str,
        user_agent: str = "Mozilla/5.0",
        timeout: int = 10,
        max_retries: int = 3,
        throttle_seconds: float = 1.0,
    ):
        self.base_url = base_url
        self.timeout = timeout
        self.throttle_seconds = throttle_seconds
        self.last_request_time = 0.0
        self.robots_checker = RobotsTxtChecker(user_agent)

        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})

        # Configure retry strategy
        retry_strategy = Retry(
            total=max_retries,
            status_forcelist=[429, 500, 502, 503, 504],
            method_whitelist=["GET", "HEAD"],
            backoff_factor=1,
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def _throttle(self) -> None:
        """Enforce throttling between requests"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.throttle_seconds:
            time.sleep(self.throttle_seconds - elapsed)
        self.last_request_time = time.time()

    def fetch(self, url: str) -> Optional[str]:
        """Fetch a single URL with throttling and robots.txt check"""
        if not self.robots_checker.can_fetch(url):
            logger.warning(f"Blocked by robots.txt: {url}")
            return None

        self._throttle()

        try:
            resp = self.session.get(url, timeout=self.timeout)
            resp.raise_for_status()
            logger.info(f"Fetched: {url} ({len(resp.content)} bytes)")
            return resp.text
        except requests.RequestException as e:
            logger.error(f"Failed to fetch {url}: {e}")
            return None

    def parse_articles(self, html: str) -> List[Dict[str, str]]:
        """Parse article list from HTML"""
        soup = BeautifulSoup(html, "html.parser")
        articles = []

        for article_elem in soup.select("article.post, div.post-item"):
            try:
                title_elem = article_elem.select_one("h2 a, h3 a, a.post-title")
                title = title_elem.get_text(strip=True) if title_elem else "N/A"

                link_elem = article_elem.select_one("a[href]")
                link = link_elem.get("href") if link_elem else "#"
                link = urljoin(self.base_url, link)

                excerpt_elem = article_elem.select_one("p, .excerpt, .summary")
                excerpt = excerpt_elem.get_text(strip=True)[:200] if excerpt_elem else ""

                date_elem = article_elem.select_one("time, .date, .published")
                date_str = date_elem.get_text(strip=True) if date_elem else "Unknown"

                articles.append({
                    "title": title,
                    "url": link,
                    "excerpt": excerpt,
                    "date": date_str,
                })
            except Exception as e:
                logger.warning(f"Failed to parse article element: {e}")
                continue

        return articles

    def paginate(
        self,
        start_url: str,
        next_selector: str = "a.next, a[rel='next']",
        max_pages: int = 5,
    ) -> Generator[List[Dict[str, str]], None, None]:
        """Paginate through results"""
        current_url = urljoin(self.base_url, start_url)
        page_count = 0

        while current_url and page_count < max_pages:
            html = self.fetch(current_url)
            if not html:
                break

            articles = self.parse_articles(html)
            yield articles

            # Find next page link
            soup = BeautifulSoup(html, "html.parser")
            next_link = soup.select_one(next_selector)
            if next_link and next_link.get("href"):
                current_url = urljoin(current_url, next_link.get("href"))
                page_count += 1
            else:
                break

    def scrape_to_json(self, start_url: str, output_file: Path) -> None:
        """Scrape multiple pages and save to JSON"""
        all_articles = []

        for page_articles in self.paginate(start_url):
            all_articles.extend(page_articles)
            logger.info(f"Collected {len(all_articles)} articles so far...")

        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_articles, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(all_articles)} articles to {output_file}")


def main():
    """Example usage"""
    scraper = WebScraper(
        base_url="https://example-blog.com",
        user_agent="WiseScraper/1.0",
        throttle_seconds=2.0,
        max_retries=2,
    )

    # Single page fetch and parse
    html = scraper.fetch("https://example-blog.com/articles")
    if html:
        articles = scraper.parse_articles(html)
        print(f"Found {len(articles)} articles on first page")
        for article in articles[:3]:
            print(f"  - {article['title']} ({article['date']})")

    # Multi-page scrape
    output = Path("articles.json")
    scraper.scrape_to_json("/articles", output)


if __name__ == "__main__":
    main()
