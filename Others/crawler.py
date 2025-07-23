import os, time, requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from requests.exceptions import ReadTimeout, HTTPError

PDF_FOLDER = "pdfs"
HEADERS = {"User-Agent": "ESG-Ingestion/1.0"}
BASE_DOMAIN = "https://www.sustainability-reports.com"
SEED_INDEX = BASE_DOMAIN + "/annual-reports/"

os.makedirs(PDF_FOLDER, exist_ok=True)

def fetch_html(url: str) -> str:
    r = requests.get(url, headers=HEADERS, timeout=10)
    r.raise_for_status()
    return r.text


def download_pdf(url: str, dest: str, retries=3):
    for i in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            r.raise_for_status()
            with open(dest, "wb") as f: f.write(r.content)
            return
        except (ReadTimeout, HTTPError):
            time.sleep(2 ** i)
    print(f"Failed to download: {url}")


def crawl_pdfs():
    html = fetch_html(SEED_INDEX)
    soup = BeautifulSoup(html, "html.parser")
    links = [urljoin(BASE_DOMAIN, a['href']) for a in soup.select("a[href*='annual']")]
    for page in links:
        sub = fetch_html(page)
        sub_soup = BeautifulSoup(sub, "html.parser")
        for a in sub_soup.find_all('a', href=True):
            href = urljoin(page, a['href'])
            if href.lower().endswith('.pdf'):
                fname = os.path.join(PDF_FOLDER, href.split('/')[-1])
                if not os.path.exists(fname): download_pdf(href, fname)