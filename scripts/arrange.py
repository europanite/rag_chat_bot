import os, re, json, hashlib
from pathlib import Path
from datetime import datetime, timezone

from PIL import Image, ImageFilter, ImageEnhance, ImageOps

LATEST_PATH = Path(os.environ.get("LATEST_PATH", "frontend/app/public/latest.json"))
FEED_DIR = Path(os.environ.get("FEED_DIR", "frontend/app/public/feed"))
OUT_DIR = Path(os.environ.get("OUT_DIR", "frontend/app/public/image"))

PAGE_W = int(os.environ.get("PAGE_W", "640"))
PAGE_H = int(os.environ.get("PAGE_H", "480"))

INPUT_IMAGE = Path(os.environ.get("INPUT_IMAGE", "snapshot.jpg"))
MODE = os.environ.get("MODE", "pillow").strip()          # pillow | img2img
STYLE = os.environ.get("STYLE", "grade").strip()         # grade | oil | watercolor | photo
STEPS = int(os.environ.get("STEPS", "2"))
STRENGTH = float(os.environ.get("STRENGTH", "0.45"))
GUIDANCE_SCALE = float(os.environ.get("GUIDANCE_SCALE", "0.0"))

MODEL_ID = os.environ.get("MODEL_ID", "stabilityai/sd-turbo").strip()
PLACE = os.environ.get("PLACE", "").strip()

PROMPT = os.environ.get("PROMPT", "").strip()
NEGATIVE = os.environ.get("NEGATIVE", "").strip()

PATCH_JSON = (os.environ.get("PATCH_JSON", "1").strip() != "0")  # dont update latest/feed if 0
OUT_NAME = os.environ.get("OUT_NAME", "").strip()


def load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def dump_json(p: Path, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def clean_for_prompt(text: str) -> str:
    t = re.sub(r"https?://\S+", "", text)
    t = re.sub(r"#\w+", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t[:220]


def slug(s: str) -> str:
    s = re.sub(r"\s+", "_", s.strip())
    s = re.sub(r"[^0-9A-Za-z_\-]", "", s)
    return s[:60] or "item"


def fit_cover(img: Image.Image, w: int, h: int) -> Image.Image:
    img = img.convert("RGB")
    sw, sh = img.size
    scale = max(w / sw, h / sh)
    nw, nh = int(sw * scale + 0.5), int(sh * scale + 0.5)
    img = img.resize((nw, nh), Image.LANCZOS)
    left = max(0, (nw - w) // 2)
    top = max(0, (nh - h) // 2)
    return img.crop((left, top, left + w, top + h))


def pillow_arrange(img: Image.Image, style: str) -> Image.Image:
    img = img.convert("RGB")

    if style in ("photo", "grade"):
        img = ImageOps.autocontrast(img, cutoff=1)
        img = ImageEnhance.Color(img).enhance(1.10)
        img = ImageEnhance.Contrast(img).enhance(1.08)
        img = ImageEnhance.Sharpness(img).enhance(1.20)
        return img

    if style == "watercolor":
        img = ImageOps.autocontrast(img, cutoff=1)
        img = img.filter(ImageFilter.GaussianBlur(radius=1.0))
        img = ImageEnhance.Color(img).enhance(0.95)
        img = ImageEnhance.Brightness(img).enhance(1.03)
        img = ImageOps.posterize(img, bits=5)
        img = img.filter(ImageFilter.SMOOTH)
        return img

    if style == "oil":
        img = ImageOps.autocontrast(img, cutoff=1)
        img = img.filter(ImageFilter.ModeFilter(size=7))
        img = ImageFilter.SMOOTH_MORE.filter(img)
        img = ImageOps.posterize(img, bits=4)
        img = img.filter(ImageFilter.EDGE_ENHANCE_MORE)
        return img

    # fallback
    return img


def img2img_arrange(base: Image.Image, prompt: str, negative: str, seed: int) -> Image.Image:
    import torch
    from diffusers import AutoPipelineForImage2Image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = AutoPipelineForImage2Image.from_pretrained(MODEL_ID, torch_dtype=dtype).to(device)

    try:
        pipe.set_progress_bar_config(disable=True)
    except Exception:
        pass
    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass
    try:
        pipe.enable_vae_slicing()
    except Exception:
        pass

    g = torch.Generator(device=device).manual_seed(seed)

    out = pipe(
        prompt=prompt,
        negative_prompt=negative,
        image=base,
        strength=max(0.05, min(0.95, STRENGTH)),
        guidance_scale=GUIDANCE_SCALE,
        num_inference_steps=max(1, min(4, STEPS)),
        generator=g,
    ).images[0].convert("RGB")

    return out


def patch_feed_file(p: Path, *, date: str, text: str, generated_at: str, feed_stem: str, rel_url: str, prompt: str, now_iso: str) -> bool:
    try:
        obj = load_json(p)
    except Exception:
        return False
    if not isinstance(obj, dict) or not isinstance(obj.get("items"), list):
        return False

    changed = False
    for it in obj["items"]:
        if not isinstance(it, dict):
            continue
        same_id = bool(generated_at) and str(it.get("id", "")) == generated_at
        same_dt = str(it.get("date", "")) == date and str(it.get("text", "")) == text
        if same_id or same_dt:
            it["id"] = feed_stem
            it["image"] = rel_url
            it["image_url"] = rel_url
            it["image_prompt"] = prompt
            it["image_model"] = MODEL_ID
            it["image_generated_at"] = now_iso
            changed = True

    if changed:
        dump_json(p, obj)
    return changed


def main() -> int:
    if not INPUT_IMAGE.exists():
        raise SystemExit(f"INPUT_IMAGE not found: {INPUT_IMAGE}")

    latest = {}
    if LATEST_PATH.exists():
        latest = load_json(LATEST_PATH)

    # latest.json fallback
    date = str(latest.get("date", "unknown")).strip()
    text = str(latest.get("text", "snapshot-based image")).strip()
    place = str(latest.get("place", "") or PLACE).strip()
    generated_at = str(latest.get("generated_at", "")).strip()

    seed_src = f"{date}\n{place}\n{generated_at}\n{text}".encode("utf-8")
    seed_hex8 = hashlib.sha1(seed_src).hexdigest()[:8]
    seed = int(seed_hex8, 16)

    feeds = []
    if FEED_DIR.exists():
        feeds = sorted(FEED_DIR.glob("feed_*.json"), key=lambda x: x.name, reverse=True)

    feed_stem = feeds[0].stem if feeds else f"{slug(date)}_{seed_hex8}"

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if OUT_NAME:
        fn = OUT_NAME
        feed_stem = Path(OUT_NAME).stem
    else:
        fn = f"{feed_stem}.png"

    out_path = OUT_DIR / fn

    base = Image.open(INPUT_IMAGE).convert("RGB")
    base = fit_cover(base, PAGE_W, PAGE_H)

    core = clean_for_prompt(text)

    if not PROMPT:
        # realistic
        PROMPT_LOCAL = (
            "photorealistic photo, natural colors, documentary style, "
            f"winter scene in {place or 'Yokosuka, Japan'}, "
            "same scene and composition as the input image. "
            f"Inspired by: {core}"
        )
    else:
        PROMPT_LOCAL = PROMPT

    if not NEGATIVE:
        NEGATIVE_LOCAL = (
            "anime, illustration, cartoon, CGI, fantasy, unreal, "
            "text, watermark, logo, letters, words, typography, caption, subtitles, signature, "
            "low quality, blurry"
        )
    else:
        NEGATIVE_LOCAL = NEGATIVE

    if MODE == "img2img":
        arranged = img2img_arrange(base, PROMPT_LOCAL, NEGATIVE_LOCAL, seed=seed)
    else:
        arranged = pillow_arrange(base, STYLE)

    arranged.save(out_path, format="PNG", optimize=True)

    # JSON PATCH_JSON=0
    if PATCH_JSON:
        now_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

        rel_url = ""
        try:
            rel_url = "/" + str(out_path.relative_to(Path("frontend/app/public"))).replace("\\", "/")
        except Exception:
            # OUT_DIR
            rel_url = ""

        if rel_url:
            latest["image_url"] = rel_url
            latest["image_prompt"] = PROMPT_LOCAL if MODE == "img2img" else f"pillow:{STYLE}"
            latest["image_model"] = MODEL_ID if MODE == "img2img" else "pillow"
            latest["image_generated_at"] = now_iso
            dump_json(LATEST_PATH, latest)

            if FEED_DIR.exists() and feeds:
                patch_feed_file(
                    feeds[0],
                    date=date,
                    text=text,
                    generated_at=generated_at,
                    feed_stem=feed_stem,
                    rel_url=rel_url,
                    prompt=latest["image_prompt"],
                    now_iso=now_iso,
                )

    print("Generated:", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
