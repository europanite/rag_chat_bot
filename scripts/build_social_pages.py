#!/usr/bin/env python3
import os, json, re
from html import escape as html_escape
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
    rel2 = rel
    while rel2.startswith("./"):
        rel2 = rel2[2:]
    while rel2.startswith("../"):
        rel2 = rel2[3:]
    rel2 = rel2.lstrip("/")
    root = f"{site}{base_path}".rstrip("/")
    return f"{root}/{rel2}" if rel2 else f"{root}/"


def xml_text(s: str) -> str:
    return html_escape(s or "", quote=False)


def article_sitemap_urls(public_dir: Path, site: str, base_path: str):
    articles_dir = public_dir / "articles"
    seen: set[str] = set()
    urls = []

    index_url = abs_url(site, base_path, "articles/index.html")
    urls.append((index_url, None))
    seen.add(index_url)

    for path in sorted(articles_dir.glob("*.json")):
        if path.name in {"featured.json", "index.json"}:
            continue
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue

        permalink = str(obj.get("permalink") or "").strip()
        if not permalink:
            continue
        loc = abs_url(site, base_path, permalink)
        if not loc or loc in seen:
            continue

        lastmod = None
        published_at = str(obj.get("published_at") or "").strip()
        if published_at:
            try:
                lastmod = datetime.fromisoformat(published_at.replace("Z", "+00:00")).date().isoformat()
            except Exception:
                lastmod = None

        urls.append((loc, lastmod))
        seen.add(loc)

    return urls

def main() -> int:
    public_dir = Path(os.environ.get("PUBLIC_DIR", "frontend/app/public"))
    feed_dir = Path(os.environ.get("FEED_DIR", str(public_dir / "feed")))
    out_dir = Path(os.environ.get("OUT_DIR", str(public_dir / "p")))
    limit = int(os.environ.get("SOCIAL_PAGE_LIMIT", "300"))

    site = (os.environ.get("SITE_URL", "") or "").rstrip("/")
    custom_domain = (os.environ.get("CUSTOM_DOMAIN", "") or "").strip().strip("/")
    if not site and custom_domain:
        if custom_domain.startswith(("http://", "https://")):
            site = custom_domain.rstrip("/")
        else:
            site = f"https://{custom_domain}"

    base_path = (os.environ.get("BASE_PATH", "") or "").rstrip("/")
    if not base_path.startswith("/") and base_path:
        base_path = "/" + base_path

    files = sorted(feed_dir.glob("feed_*.json"), reverse=True)[:limit]
    ga_measurement_id = (os.environ.get("EXPO_PUBLIC_GA_MEASUREMENT_ID", "") or "").strip()

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
        share_url = f"{site}{base_path}/p/{pid}/index.html" if site else f"./p/{pid}/index.html"
        app_url = f"{site}{base_path}/?post={pid}" if site else f"./?post={pid}"

        share_path.parent.mkdir(parents=True, exist_ok=True)
        ga_tag = ""
        if ga_measurement_id:
            ga_tag = f"""
  <script async src="https://www.googletagmanager.com/gtag/js?id={ga_measurement_id}"></script>
  <script>
    window.dataLayer = window.dataLayer || [];
    function gtag(){{dataLayer.push(arguments);}}
    gtag('js', new Date());
    gtag('config', '{ga_measurement_id}', {{
      page_location: '{share_url}',
      page_path: '/p/{pid}/index.html',
      page_title: 'GOODDAY YOKOSUKA'
    }});
  </script>"""
        html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>GOODDAY YOKOSUKA</title>
  <meta name="description" content="{desc}" />
  <meta name="robots" content="noindex,follow,max-image-preview:large" />
  <meta name="googlebot" content="noindex,follow,max-image-preview:large" />
  <link rel="canonical" href="{app_url}" />
  <meta property="og:type" content="article" />
  <meta property="og:site_name" content="GOODDAY YOKOSUKA" />
  <meta property="og:title" content="GOODDAY YOKOSUKA" />
  <meta property="og:description" content="{desc}" />
  <meta property="og:url" content="{share_url}" />
  {"<meta property=\"og:image\" content=\"" + og_image + "\" />" if og_image else ""}
  <meta name="twitter:card" content="summary_large_image" />
  {ga_tag}
  <meta http-equiv="refresh" content="0;url={app_url}" />
</head>
<body>
  <p>Redirecting… <a href="{app_url}">Open post</a></p>
</body>
</html>
"""
        share_path.write_text(html, encoding="utf-8")


    # robots.txt + sitemap.xml
    if site:
        urls_for_sitemap.extend(article_sitemap_urls(public_dir, site, base_path))

        deduped_urls = []
        seen = set()
        for loc, lm in urls_for_sitemap:
            if not loc or loc in seen:
                continue
            deduped_urls.append((loc, lm))
            seen.add(loc)

        robots = f"User-agent: *\nAllow: /\nSitemap: {site}{base_path}/sitemap.xml\n"
        (public_dir / "robots.txt").write_text(robots, encoding="utf-8")

        items = []
        for loc, lm in deduped_urls:
            if lm:
                items.append(f"<url><loc>{xml_text(loc)}</loc><lastmod>{xml_text(lm)}</lastmod></url>")
            else:
                items.append(f"<url><loc>{xml_text(loc)}</loc></url>")
        sitemap = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n" \
                  "<urlset xmlns=\"http://www.sitemaps.org/schemas/sitemap/0.9\">\n" + \
                  "\n".join(items) + "\n</urlset>\n"
        (public_dir / "sitemap.xml").write_text(sitemap, encoding="utf-8")

    print(f"Generated share pages: {len(files)}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
