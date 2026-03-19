#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

try:
    from artifact_paths import artifact_public_dir
except ImportError:
    from scripts.artifact_paths import artifact_public_dir

try:
    from prepare_drive_article_teaser import (
        FolderRef,
        build_article_html,
        build_articles_index_html,
        choose_article_doc,
        choose_default_hero_image,
        download_entry_bytes,
        download_ref,
        download_text,
        ensure_iso,
        extract_folder_ref,
        list_public_folder_entries,
        parse_article_markdown,
        save_article_image,
        slugify,
        utc_date_from_iso,
        write_json,
    )
except ImportError:
    from scripts.prepare_drive_article_teaser import (
        FolderRef,
        build_article_html,
        build_articles_index_html,
        choose_article_doc,
        choose_default_hero_image,
        download_entry_bytes,
        download_ref,
        download_text,
        ensure_iso,
        extract_folder_ref,
        list_public_folder_entries,
        parse_article_markdown,
        save_article_image,
        slugify,
        utc_date_from_iso,
        write_json,
    )


def eprint(*args: Any) -> None:
    print(*args, file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Enumerate all child folders under a public Google Drive article root and "
            "build frontend/app/public/articles/index.json for the web left sidebar."
        )
    )
    p.add_argument("--root-folder-url", required=True, help="Public Google Drive root folder URL or folder ID")
    p.add_argument("--place", default="Yokosuka", help="Fallback place label")
    p.add_argument("--public-dir", default="", help="Defaults to artifact_public_dir()/frontend/app/public")
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


def main() -> int:
    args = build_parser().parse_args()
    public_dir = Path(args.public_dir).resolve() if args.public_dir else artifact_public_dir().resolve()
    articles_dir = public_dir / "articles"
    root = extract_folder_ref(args.root_folder_url)

    root_entries = list_public_folder_entries(root, include_folders=True)
    child_folders = [entry for entry in root_entries if entry.kind == "folder"]
    if not child_folders:
        raise SystemExit("No child folders were found under the articles root folder")

    used_slugs: set[str] = set()
    built_articles: List[Dict[str, Any]] = []

    for folder in child_folders:
        try:
            folder_ref = FolderRef(folder_id=folder.file_id, resource_key=folder.resource_key)
            folder_entries = list_public_folder_entries(folder_ref, include_folders=False)
            article_doc = choose_article_doc(folder_entries)
            parsed = parse_article_markdown(download_text(article_doc))

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
                hero_url = save_article_image(public_dir, slug, hero_name, hero_bytes, hero_mime)
            else:
                default_hero = choose_default_hero_image(folder_entries)
                if default_hero is not None:
                    hero_bytes, hero_mime = download_entry_bytes(default_hero)
                    hero_url = save_article_image(public_dir, slug, default_hero.name, hero_bytes, hero_mime)

            photos: List[Dict[str, str]] = []
            for idx, photo_ref in enumerate(parsed.get("photos") or []):
                photo_bytes, photo_mime, photo_name = download_ref(photo_ref, folder_entries)
                photo_url = save_article_image(public_dir, slug, f"{idx + 1:02d}_{photo_name}", photo_bytes, photo_mime)
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

            permalink = f"./articles/{slug}/index.html"
            guides_permalink = "./articles/index.html"
            article_json: Dict[str, Any] = {
                "id": folder.file_id,
                "slug": slug,
                "title": str(parsed.get("title") or folder.name),
                "short_title": str(parsed.get("short_title") or parsed.get("title") or folder.name),
                "category": str(parsed.get("category") or "article"),
                "place": place,
                "published_at": published_at,
                "date": published_date,
                "summary": str(parsed.get("summary") or ""),
                "short_text": str(parsed.get("short_text") or parsed.get("summary") or ""),
                "hero_image": hero_url,
                "photos": photos,
                "links": links,
                "body_md": str(parsed.get("body_md") or ""),
                "permalink": permalink,
                "guides_permalink": guides_permalink,
                "source_article_folder_id": folder.file_id,
                "source_article_doc_id": article_doc.file_id,
                "source_article_folder_url": folder.href,
                "source_article_doc_url": article_doc.href,
            }

            write_json(articles_dir / f"{slug}.json", article_json)
            article_page_dir = articles_dir / slug
            article_page_dir.mkdir(parents=True, exist_ok=True)
            (article_page_dir / "index.html").write_text(
                build_article_html(article_json, guides_href="../index.html", feed_href="../../"),
                encoding="utf-8",
            )
            built_articles.append(article_json)
            print(f"Built guide: {article_json['title']} ({folder.name})")
        except SystemExit as exc:
            eprint(f"WARN: skipped folder {folder.name!r}: {exc}")
        except Exception as exc:
            eprint(f"WARN: skipped folder {folder.name!r}: {exc}")

    if not built_articles:
        raise SystemExit("No guides could be built from the public Google Drive root folder")

    built_articles.sort(key=article_sort_key, reverse=True)
    write_json(articles_dir / "featured.json", built_articles[0])

    (articles_dir / "index.html").write_text(
        build_articles_index_html(built_articles, feed_href="../"),
        encoding="utf-8",
    )
    write_json(
        articles_dir / "index.json",
        {
            "updated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "items": [
                {
                    "id": str(item.get("id") or item.get("slug") or ""),
                    "slug": str(item.get("slug") or ""),
                    "title": str(item.get("title") or ""),
                    "summary": str(item.get("summary") or ""),
                    "place": str(item.get("place") or ""),
                    "published_at": str(item.get("published_at") or ""),
                    "date": str(item.get("date") or ""),
                    "hero_image": str(item.get("hero_image") or ""),
                    "permalink": str(item.get("permalink") or ""),
                }
                for item in built_articles
            ],
        },
    )

    print(f"Wrote guide index JSON: {articles_dir / 'index.json'} ({len(built_articles)} guides)")
    print(f"Wrote guide index HTML: {articles_dir / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
