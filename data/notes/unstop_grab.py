"""Attach to the user's logged-in Chrome (CDP :9222), open the ML round, and save everything it shows.

Saves: every unstop API JSON the page loads, the rendered text, all links, a screenshot.
Never clicks anything -- fullscreen proctoring counts violations, so the user drives any "Start" button.
"""
import json, re, sys, time
from pathlib import Path
from playwright.sync_api import sync_playwright

OUT = Path(__file__).parent / "grab"
OUT.mkdir(exist_ok=True)
URL = "https://unstop.com/competitions/1743604/round/1593683/play/code"
navigate = "--no-nav" not in sys.argv

with sync_playwright() as p:
    browser = p.chromium.connect_over_cdp("http://127.0.0.1:9222")
    ctx = browser.contexts[0]
    pages = [pg for pg in ctx.pages if "unstop.com" in pg.url] or ctx.pages
    page = pages[-1]

    n = 0
    def on_response(r):
        global n
        if "unstop.com/api" not in r.url or r.request.resource_type not in ("xhr", "fetch"):
            return
        try:
            body = r.json()
        except Exception:
            return
        n += 1
        name = re.sub(r"[^A-Za-z0-9]+", "_", r.url.split("unstop.com/")[1])[:120]
        (OUT / f"api_{n:03d}_{name}.json").write_text(json.dumps({"url": r.url, "status": r.status, "body": body}, indent=1))
    page.on("response", on_response)

    page.set_viewport_size({"width": 1440, "height": 900})  # site refuses <1024px
    if navigate:
        page.goto(URL, wait_until="domcontentloaded")
    try:
        page.wait_for_load_state("networkidle", timeout=25000)
    except Exception:
        pass
    time.sleep(3)

    (OUT / "page_url.txt").write_text(page.url)
    (OUT / "page_text.txt").write_text(page.inner_text("body"))
    (OUT / "page.html").write_text(page.content())
    links = page.eval_on_selector_all("a[href]", "els => els.map(e => [e.innerText.trim(), e.href])")
    (OUT / "links.json").write_text(json.dumps(links, indent=1))
    page.screenshot(path=str(OUT / "page.png"), full_page=True)
    print("url:", page.url, "| api responses:", n, "| links:", len(links))
    # no browser.close(): the user's Chrome must stay open and logged in
