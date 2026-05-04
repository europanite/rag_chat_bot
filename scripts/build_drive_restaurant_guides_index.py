#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

try:
    from artifact_paths import artifact_public_dir, resolve_public_image_url
except ImportError:
    from scripts.artifact_paths import artifact_public_dir, resolve_public_image_url

try:
    from prepare_drive_article_teaser import (
        FolderRef,
        absolute_site_url,
        choose_article_doc,
        choose_default_hero_image,
        download_entry_bytes,
        download_ref,
        download_text,
        ensure_iso,
        extract_folder_ref,
        ga4_head_snippet,
        infer_ext,
        list_public_folder_entries,
        markdown_to_html,
        parse_article_source,
        rebase_public_url,
        sanitize_filename,
        slugify,
        utc_date_from_iso,
        write_json,
        IMAGE_EXTS,
    )
except ImportError:
    from scripts.prepare_drive_article_teaser import (
        FolderRef,
        absolute_site_url,
        choose_article_doc,
        choose_default_hero_image,
        download_entry_bytes,
        download_ref,
        download_text,
        ensure_iso,
        extract_folder_ref,
        ga4_head_snippet,
        infer_ext,
        list_public_folder_entries,
        markdown_to_html,
        parse_article_source,
        rebase_public_url,
        sanitize_filename,
        slugify,
        utc_date_from_iso,
        write_json,
        IMAGE_EXTS,
    )


DEFAULT_SECTION = "restaurant_articles"
DEFAULT_SECTION_LABEL = "Restaurant Guide"


def eprint(*args: Any) -> None:
    print(*args, file=sys.stderr)


def clean_section(raw: str) -> str:
    section = str(raw or "").strip().strip("/")
    if not section:
        raise SystemExit("section must not be empty")
    if "/" in section or "\\" in section or section.startswith("."):
        raise SystemExit(f"invalid section path: {raw!r}")
    return section


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Enumerate child folders under a public Google Drive restaurant article root and "
            "build frontend/app/public/restaurant_articles/index.json and static HTML pages."
        )
    )
    p.add_argument("--root-folder-url", required=True, help="Public Google Drive root folder URL or folder ID")
    p.add_argument("--place", default="Yokosuka", help="Fallback place label")
    p.add_argument("--public-dir", default="", help="Defaults to artifact_public_dir()/frontend/app/public")
    p.add_argument("--section", default=DEFAULT_SECTION, help="Output section directory. Default: restaurant_articles")
    p.add_argument("--section-label", default=DEFAULT_SECTION_LABEL, help="Human-readable label shown in generated HTML")
    return p


def make_unique_slug(raw_slug: str, used: set[str], folder_id: str) -> str:
    base = slugify(raw_slug)
    if base not in used:
        used.add(base)
        return base

    suffix = (folder_id or "")[:8].lower() or "dup"
    candidate = f"{base}-{suffix}"
    if candidate not in used:
        used.add(candidate)
        return candidate

    i = 2
    while True:
        candidate = f"{base}-{suffix}-{i}"
        if candidate not in used:
            used.add(candidate)
            return candidate
        i += 1


def article_sort_key(item: Dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(item.get("published_at") or ""),
        str(item.get("date") or ""),
        str(item.get("title") or "").casefold(),
    )


def save_section_image(public_dir: Path, section: str, slug: str, filename: str, data: bytes, mime_type: str) -> str:
    ext = infer_ext(filename, mime_type, default=".bin")
    if ext.lower() not in IMAGE_EXTS:
        raise SystemExit(f"Downloaded asset is not an image: {filename!r} mime={mime_type!r}")
    base = sanitize_filename(Path(filename).stem)
    final_name = f"{base}{ext.lower()}"
    rel = Path("image") / section / slug / final_name
    out_path = public_dir / rel
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(data)
    return resolve_public_image_url(str(rel).replace("\\", "/"))


def build_section_article_html(
    article: Dict[str, Any],
    *,
    section: str,
    section_label: str,
    index_href: str = "../index.html",
    feed_href: str = "../../",
) -> str:
    title = html.escape(str(article.get("title") or "GOODDAY YOKOSUKA"))
    summary = html.escape(str(article.get("summary") or ""))
    place = html.escape(str(article.get("place") or "Yokosuka"))
    published_at = html.escape(str(article.get("published_at") or ""))
    hero = html.escape(rebase_public_url(str(article.get("hero_image") or ""), levels_up=2))
    body_html = markdown_to_html(str(article.get("body_md") or ""))
    index_href = html.escape(index_href)
    feed_href = html.escape(feed_href)
    slug = slugify(str(article.get("slug") or article.get("title") or "article"))
    article_path = f"{section}/{slug}/index.html"
    canonical_url = html.escape(absolute_site_url(article_path))
    hero_abs = html.escape(absolute_site_url(str(article.get("hero_image") or "")))
    page_path = f"/{section}/{slug}/index.html"
    ga_tag = ga4_head_snippet(page_title=f"{title} | GOODDAY YOKOSUKA", page_path=page_path)
    section_label_esc = html.escape(section_label)

    photo_html: list[str] = []
    for photo in article.get("photos") or []:
        url = html.escape(rebase_public_url(str(photo.get("url") or ""), levels_up=2))
        alt = html.escape(str(photo.get("alt") or ""))
        if url:
            photo_html.append(f'<figure><img src="{url}" alt="{alt}" /><figcaption>{alt}</figcaption></figure>')

    links_html: list[str] = []
    for link in article.get("links") or []:
        label = html.escape(str(link.get("title") or link.get("url") or ""))
        url = html.escape(str(link.get("url") or ""))
        if url:
            links_html.append(f'<li><a href="{url}">{label}</a></li>')

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>{title} | GOODDAY YOKOSUKA</title>
  <meta name="description" content="{summary}" />
  {f'<link rel="canonical" href="{canonical_url}" />' if canonical_url else ''}
  <meta property="og:type" content="article" />
  <meta property="og:site_name" content="GOODDAY YOKOSUKA" />
  <meta property="og:title" content="{title} | GOODDAY YOKOSUKA" />
  <meta property="og:description" content="{summary}" />
  {f'<meta property="og:url" content="{canonical_url}" />' if canonical_url else ''}
  {f'<meta property="og:image" content="{hero_abs}" />' if hero_abs else ''}
  <meta name="twitter:card" content="summary_large_image" />
  {ga_tag}
  <style>
    body {{ margin: 0; font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f6f4ff; color: #111827; }}
    .wrap {{ max-width: 860px; margin: 0 auto; padding: 24px 16px 72px; }}
    .card {{ background: #fff; border: 2px solid #000; border-radius: 20px; padding: 24px; box-shadow: 0 8px 24px rgba(0,0,0,.08); }}
    h1, h2, h3 {{ line-height: 1.25; }}
    p, li {{ line-height: 1.8; font-size: 16px; }}
    .eyebrow {{ display: inline-block; font-size: 12px; background: #fef3c7; border-radius: 999px; padding: 6px 10px; margin-bottom: 12px; }}
    .meta {{ color: #4b5563; font-size: 14px; margin-bottom: 16px; }}
    img {{ display: block; max-width: 100%; height: auto; border-radius: 16px; border: 1px solid #ddd; }}
    figure {{ margin: 20px 0; }}
    figcaption {{ color: #6b7280; font-size: 13px; margin-top: 8px; }}
    a {{ color: #0B57D0; }}
    .nav {{ display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 16px; }}
  </style>
</head>
<body>
  <main class="wrap">
    <div class="card">
      <div class="nav">
        <a href="{feed_href}">← Back to feed</a>
        <a href="{index_href}">{section_label_esc} list</a>
      </div>
      <div class="eyebrow">GOODDAY YOKOSUKA / {section_label_esc}</div>
      <h1>{title}</h1>
      <div class="meta">{place} · {published_at}</div>
      {f'<img src="{hero}" alt="{title}" />' if hero else ''}
      {f'<p>{summary}</p>' if summary else ''}
      {body_html}
      {''.join(photo_html)}
      {f'<h2>Links</h2><ul>{"".join(links_html)}</ul>' if links_html else ''}
    </div>
  </main>
</body>
</html>
"""


def build_section_index_html(
    articles: Sequence[Dict[str, Any]],
    *,
    section: str,
    section_label: str,
    feed_href: str = "../",
) -> str:
    canonical_url = html.escape(absolute_site_url(f"{section}/index.html"))
    title_text = html.escape(f"{section_label}s")
    section_label_esc = html.escape(section_label)

    cards_html: list[str] = []
    for article in articles:
        title = html.escape(str(article.get("title") or "GOODDAY YOKOSUKA"))
        summary = html.escape(str(article.get("summary") or ""))
        place = html.escape(str(article.get("place") or "Yokosuka"))
        published_at = html.escape(str(article.get("published_at") or article.get("date") or ""))
        hero = html.escape(rebase_public_url(str(article.get("hero_image") or ""), levels_up=1))
        slug = slugify(str(article.get("slug") or article.get("title") or "article"))
        article_href = html.escape(f"./{slug}/index.html")

        cards_html.append(
            f"""<article class="guide-card">
        {f'<img src="{hero}" alt="{title}" />' if hero else ''}
        <div class="eyebrow">{section_label_esc}</div>
        <h2>{title}</h2>
        <div class="meta">{place} · {published_at}</div>
        {f'<p>{summary}</p>' if summary else ''}
        <a class="cta" href="{article_href}">Open this restaurant guide →</a>
      </article>"""
        )

    feed_href = html.escape(feed_href)
    cards = "\n".join(cards_html) if cards_html else "<p>No restaurant guides yet.</p>"
    page_path = f"/{section}/index.html"
    ga_tag = ga4_head_snippet(page_title=f"{title_text} | GOODDAY YOKOSUKA", page_path=page_path)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>{title_text} | GOODDAY YOKOSUKA</title>
  <meta name="description" content="restaurant guides on GOODDAY YOKOSUKA" />
  {f'<link rel="canonical" href="{canonical_url}" />' if canonical_url else ''}
  {f'<meta property="og:url" content="{canonical_url}" />' if canonical_url else ''}
  {ga_tag}
  <style>
    body {{ margin: 0; font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f6f4ff; color: #111827; }}
    .wrap {{ max-width: 980px; margin: 0 auto; padding: 24px 16px 72px; }}
    .card {{ background: #fff; border: 2px solid #000; border-radius: 20px; padding: 24px; box-shadow: 0 8px 24px rgba(0,0,0,.08); }}
    .eyebrow {{ display: inline-block; font-size: 12px; background: #fef3c7; border-radius: 999px; padding: 6px 10px; margin-bottom: 12px; }}
    .meta {{ color: #4b5563; font-size: 14px; margin-bottom: 12px; }}
    img {{ display: block; width: 100%; max-width: 100%; height: auto; border-radius: 16px; border: 1px solid #ddd; margin-bottom: 16px; }}
    a {{ color: #0B57D0; }}
    .nav {{ display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 16px; }}
    .guides {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px; }}
    .guide-card {{ border: 1px solid #d1d5db; border-radius: 16px; padding: 16px; background: #fcfcff; }}
    .guide-card h2 {{ margin: 0 0 8px; font-size: 22px; line-height: 1.35; }}
    .guide-card p {{ line-height: 1.7; }}
    .cta {{ display: inline-block; margin-top: 8px; font-weight: 700; }}
  </style>
</head>
<body>
  <main class="wrap">
    <div class="card">
      <div class="nav"><a href="{feed_href}">← Back to feed</a></div>
      <div class="eyebrow">GOODDAY YOKOSUKA</div>
      <h1>{title_text}</h1>
      <div class="guides">
        {cards}
      </div>
    </div>
  </main>
</body>
</html>
"""


def main() -> int:
    args = build_parser().parse_args()
    public_dir = Path(args.public_dir).resolve() if args.public_dir else artifact_public_dir().resolve()
    section = clean_section(args.section)
    section_label = str(args.section_label or DEFAULT_SECTION_LABEL).strip() or DEFAULT_SECTION_LABEL
    articles_dir = public_dir / section
    root = extract_folder_ref(args.root_folder_url)

    root_entries = list_public_folder_entries(root, include_folders=True)
    child_folders = [entry for entry in root_entries if entry.kind == "folder"]
    if not child_folders:
        raise SystemExit("No child folders were found under the restaurant articles root folder")

    used_slugs: set[str] = set()
    built_articles: List[Dict[str, Any]] = []

    for folder in child_folders:
        try:
            folder_ref = FolderRef(folder_id=folder.file_id, resource_key=folder.resource_key)
            folder_entries = list_public_folder_entries(folder_ref, include_folders=False)
            article_doc = choose_article_doc(folder_entries)
            parsed = parse_article_source(article_doc, download_text(article_doc))

            slug = make_unique_slug(
                str(parsed.get("slug") or parsed.get("title") or folder.name),
                used_slugs,
                folder.file_id,
            )

            published_at = ensure_iso(str(parsed.get("published_at") or ""))
            published_date = utc_date_from_iso(published_at)
            place = str(parsed.get("place") or args.place)

            hero_url = ""
            hero_ref = parsed.get("hero_image")
            if getattr(hero_ref, "source", ""):
                hero_bytes, hero_mime, hero_name = download_ref(hero_ref, folder_entries)
                hero_url = save_section_image(public_dir, section, slug, hero_name, hero_bytes, hero_mime)
            else:
                default_hero = choose_default_hero_image(folder_entries)
                if default_hero is not None:
                    hero_bytes, hero_mime = download_entry_bytes(default_hero)
                    hero_url = save_section_image(public_dir, section, slug, default_hero.name, hero_bytes, hero_mime)

            photos: List[Dict[str, str]] = []
            for idx, photo_ref in enumerate(parsed.get("photos") or []):
                photo_bytes, photo_mime, photo_name = download_ref(photo_ref, folder_entries)
                photo_url = save_section_image(
                    public_dir,
                    section,
                    slug,
                    f"{idx + 1:02d}_{photo_name}",
                    photo_bytes,
                    photo_mime,
                )
                photos.append(
                    {
                        "url": photo_url,
                        "alt": str(getattr(photo_ref, "alt", "") or Path(photo_name).stem),
                    }
                )

            links: List[Dict[str, str]] = []
            for link_ref in parsed.get("links") or []:
                title = str(getattr(link_ref, "title", "") or getattr(link_ref, "url", "")).strip()
                url = str(getattr(link_ref, "url", "")).strip()
                if url:
                    links.append({"title": title or url, "url": url})

            permalink = f"./{section}/{slug}/index.html"
            guides_permalink = f"./{section}/index.html"
            article_json: Dict[str, Any] = {
                "id": folder.file_id,
                "slug": slug,
                "title": str(parsed.get("title") or folder.name),
                "short_title": str(parsed.get("short_title") or parsed.get("title") or folder.name),
                "category": str(parsed.get("category") or "restaurant"),
                "place": place,
                "published_at": published_at,
                "date": published_date,
                "summary": str(parsed.get("summary") or parsed.get("description") or ""),
                "description": str(parsed.get("description") or ""),
                "short_text": str(parsed.get("short_text") or parsed.get("description") or parsed.get("summary") or ""),
                "hero_image": hero_url,
                "photos": photos,
                "links": links,
                "body_md": str(parsed.get("body_md") or ""),
                "permalink": permalink,
                "guides_permalink": guides_permalink,
                "section": section,
                "source_article_folder_id": folder.file_id,
                "source_article_doc_id": article_doc.file_id,
                "source_article_folder_url": folder.href,
                "source_article_doc_url": article_doc.href,
            }

            write_json(articles_dir / f"{slug}.json", article_json)
            article_page_dir = articles_dir / slug
            article_page_dir.mkdir(parents=True, exist_ok=True)
            (article_page_dir / "index.html").write_text(
                build_section_article_html(
                    article_json,
                    section=section,
                    section_label=section_label,
                    index_href="../index.html",
                    feed_href="../../",
                ),
                encoding="utf-8",
            )
            built_articles.append(article_json)
            print(f"Built restaurant guide: {article_json['title']} ({folder.name})")
        except SystemExit as exc:
            eprint(f"WARN: skipped folder {folder.name!r}: {exc}")
        except Exception as exc:
            eprint(f"WARN: skipped folder {folder.name!r}: {exc}")

    if not built_articles:
        raise SystemExit("No restaurant guides could be built from the public Google Drive root folder")

    built_articles.sort(key=article_sort_key, reverse=True)
    write_json(articles_dir / "featured.json", built_articles[0])

    (articles_dir / "index.html").write_text(
        build_section_index_html(built_articles, section=section, section_label=section_label, feed_href="../"),
        encoding="utf-8",
    )
    write_json(
        articles_dir / "index.json",
        {
            "updated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "section": section,
            "category": "restaurant",
            "items": [
                {
                    "id": str(item.get("id") or item.get("slug") or ""),
                    "slug": str(item.get("slug") or ""),
                    "title": str(item.get("title") or ""),
                    "short_title": str(item.get("short_title") or item.get("title") or ""),
                    "summary": str(item.get("summary") or ""),
                    "description": str(item.get("description") or ""),
                    "short_text": str(item.get("short_text") or ""),
                    "category": str(item.get("category") or "restaurant"),
                    "place": str(item.get("place") or ""),
                    "published_at": str(item.get("published_at") or ""),
                    "date": str(item.get("date") or ""),
                    "hero_image": str(item.get("hero_image") or ""),
                    "permalink": str(item.get("permalink") or ""),
                    "guides_permalink": str(item.get("guides_permalink") or ""),
                    "section": section,
                }
                for item in built_articles
            ],
        },
    )

    print(f"Wrote restaurant guide index JSON: {articles_dir / 'index.json'} ({len(built_articles)} guides)")
    print(f"Wrote restaurant guide index HTML: {articles_dir / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
