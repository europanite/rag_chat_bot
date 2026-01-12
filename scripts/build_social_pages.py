#!/usr/bin/env python3
import os, json, re
from pathlib import Path
from datetime import datetime

URL_RE = re.compile(r"https?://\S+")

def strip_urls(s: str) -> str:
    s = URL_RE.sub("", s or "")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def abs_url(site: str, base_path: str, rel: str) -> str:
    if not rel:
        return ""
    if rel.startswith("http://") or rel.startswith("https://"):
        return rel
    rel2 = rel.lstrip("./")
    return f"{site}{base_path}/{rel2}"

def main() -> int:
    public_dir = Path(os.environ.get("PUBLIC_DIR", "frontend/app/public"))
    feed_dir = Path(os.environ.get("FEED_DIR", str(public_dir / "feed")))
    out_dir = Path(os.environ.get("OUT_DIR", str(public_dir / "p")))
    limit = int(os.environ.get("SOCIAL_PAGE_LIMIT", "300"))

    site = (os.environ.get("SITE_URL", "") or "").rstrip("/")
    base_path = (os.environ.get("BASE_PATH", "") or "").rstrip("/")
    if not base_path.startswith("/") and base_path:
        base_path = "/" + base_path

    files = sorted(feed_dir.glob("feed_*.json"), reverse=True)[:limit]

    urls_for_sitemap = []
    # root
    if site:
        urls_for_sitemap.append((f"{site}{base_path}/", None))

    for p in files:
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue

        pid = (obj.get("id") or p.stem or "").strip()
        if not pid:
            continue

        text = strip_urls(str(obj.get("text") or ""))
        desc = text[:220]

        gen = str(obj.get("generated_at") or "")
        lastmod = None
        try:
            # ISO-ish
            lastmod = datetime.fromisoformat(gen.replace("Z", "+00:00")).date().isoformat()
        except Exception:
            lastmod = None

        img_rel = str(obj.get("image_url") or obj.get("image") or "").strip()
        og_image = abs_url(site, base_path, img_rel) if site else ""

        share_path = out_dir / pid / "index.html"
        share_url = f"{site}{base_path}/p/{pid}/" if site else f"./p/{pid}/"
        app_url = f"{site}{base_path}/?post={pid}" if site else f"./?post={pid}"

        share_path.parent.mkdir(parents=True, exist_ok=True)
        html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>GOODDAY YOKOSUKA</title>
  <meta name="description" content="{desc}" />
  <link rel="canonical" href="{share_url}" />
  <meta property="og:type" content="article" />
  <meta property="og:site_name" content="GOODDAY YOKOSUKA" />
  <meta property="og:title" content="GOODDAY YOKOSUKA" />
  <meta property="og:description" content="{desc}" />
  <meta property="og:url" content="{share_url}" />
  {"<meta property=\"og:image\" content=\"" + og_image + "\" />" if og_image else ""}
  <meta name="twitter:card" content="summary_large_image" />
  <meta http-equiv="refresh" content="0;url={app_url}" />
</head>
<body>
  <p>Redirecting… <a href="{app_url}">Open post</a></p>
</body>
</html>
"""
        share_path.write_text(html, encoding="utf-8")

        if site:
            urls_for_sitemap.append((share_url, lastmod))

    # robots.txt + sitemap.xml
    if site:
        robots = f"User-agent: *\nAllow: /\nSitemap: {site}{base_path}/sitemap.xml\n"
        (public_dir / "robots.txt").write_text(robots, encoding="utf-8")

        items = []
        for loc, lm in urls_for_sitemap:
            if lm:
                items.append(f"<url><loc>{loc}</loc><lastmod>{lm}</lastmod></url>")
            else:
                items.append(f"<url><loc>{loc}</loc></url>")
        sitemap = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n" \
                  "<urlset xmlns=\"http://www.sitemaps.org/schemas/sitemap/0.9\">\n" + \
                  "\n".join(items) + "\n</urlset>\n"
        (public_dir / "sitemap.xml").write_text(sitemap, encoding="utf-8")

    print(f"Generated share pages: {len(files)}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
