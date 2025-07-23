# crawler.py

This script automates the discovery and downloading of annual report PDFs from the Sustainability Reports website.

## Features

- Fetches the annual reports index page to find links to individual report pages.
- Parses each report page to locate all PDF links.
- Downloads each PDF into a local `pdfs/` directory, retrying on transient errors.

## Requirements

- Python 3.7 or higher
- `requests`
- `beautifulsoup4`

Install dependencies via pip:

```bash
pip install requests beautifulsoup4
