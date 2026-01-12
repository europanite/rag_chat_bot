import os, re, json, hashlib
from pathlib import Path
from datetime import datetime, timezone
from typing import Any
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

# img2img CPU tuning / overrides
DEVICE = os.environ.get("DEVICE", "auto").strip().lower()          # auto | cpu | cuda
TORCH_NUM_THREADS = int(os.environ.get("TORCH_NUM_THREADS", "0"))  # 0 = leave default
SEED_OVERRIDE = os.environ.get("SEED", "").strip()                 # optional int
SEED_OFFSET = int(os.environ.get("SEED_OFFSET", "0"))              # add to computed seed

# optional: run SD at smaller size for CPU speed, then upscale to PAGE_W/H
SD_W = int(os.environ.get("SD_W", "0"))
SD_H = int(os.environ.get("SD_H", "0"))

MODEL_ID = os.environ.get("MODEL_ID", "stabilityai/sd-turbo").strip()
PLACE = os.environ.get("PLACE", "").strip()

PROMPT = os.environ.get("PROMPT", "").strip()
NEGATIVE = os.environ.get("NEGATIVE", "").strip()

MIRROR_AUG = (os.environ.get("MIRROR_AUG", "0").strip() != "0")  # mirror->img2img->mirror back (double flip)

PATCH_JSON = (os.environ.get("PATCH_JSON", "1").strip() != "0")  # dont update latest/feed if 0
OUT_NAME = os.environ.get("OUT_NAME", "").strip()


def load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def dump_json(p: Path, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def clean_for_prompt(text: str) -> str:
    t = re.sub(r"https?://\S+", "", text)
    t = re.sub(r"\s+", " ", t).strip()
    # keep it short-ish for stable diffusion
    return t[:220]



def _clip_safe_prompt(pipe: Any, text: str) -> str:
    """
    Truncate prompt to the CLIP tokenizer max length (typically 77 tokens),
    to avoid: 'Token indices sequence length is longer than ... (80 > 77)'.
    """
    text = (text or "").strip()
    if not text:
        return text
    try:
        tok = getattr(pipe, "tokenizer", None)
        if tok is None:
            return text
        max_len = int(getattr(tok, "model_max_length", 77) or 77)
        enc = tok(text, truncation=True, max_length=max_len, return_tensors=None)
        ids = enc.get("input_ids")
        # tokenizer returns either List[int] or List[List[int]]
        if isinstance(ids, list) and ids and isinstance(ids[0], list):
            ids = ids[0]
        if not isinstance(ids, list) or not ids:
            return text
        return tok.decode(ids, skip_special_tokens=True).strip()
    except Exception:
        return text


def _from_pretrained_img2img(AutoPipelineForImage2Image: Any, model_id: str, dtype: Any, **kwargs: Any):
    """Prefer new 'dtype=' API, fallback to 'torch_dtype=' for older diffusers."""
    try:
        return AutoPipelineForImage2Image.from_pretrained(model_id, dtype=dtype, **kwargs)
    except TypeError:
        return AutoPipelineForImage2Image.from_pretrained(model_id, torch_dtype=dtype, **kwargs)


def slug(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s or "x"


def _floor_to_8(x: int) -> int:
    # keep SD sizes valid (multiple of 8)
    x = int(x)
    if x <= 0:
        return 0
    return max(64, (x // 8) * 8)

def fit_cover(img: Image.Image, w: int, h: int) -> Image.Image:
    iw, ih = img.size
    if iw <= 0 or ih <= 0:
        return img.resize((w, h))
    scale = max(w / iw, h / ih)
    nw, nh = int(iw * scale + 0.5), int(ih * scale + 0.5)
    img = img.resize((nw, nh), Image.Resampling.LANCZOS)
    left = max(0, (nw - w) // 2)
    top = max(0, (nh - h) // 2)
    return img.crop((left, top, left + w, top + h))


def pillow_arrange(img: Image.Image, style: str) -> Image.Image:
    style = (style or "grade").strip().lower()

    if style == "oil":
        # soft "oil-ish" look: blur + contrast + slight edge enhance
        base = img.filter(ImageFilter.GaussianBlur(radius=1.2))
        base = ImageEnhance.Color(base).enhance(1.15)
        base = ImageEnhance.Contrast(base).enhance(1.18)
        base = base.filter(ImageFilter.EDGE_ENHANCE_MORE)
        return base

    if style == "watercolor":
        # watercolor-ish: soften + brighten + reduce contrast a bit
        base = img.filter(ImageFilter.GaussianBlur(radius=1.8))
        base = ImageEnhance.Brightness(base).enhance(1.05)
        base = ImageEnhance.Contrast(base).enhance(0.95)
        base = ImageEnhance.Color(base).enhance(1.10)
        return base

    if style == "photo":
        # light photo grading: sharpen + mild contrast
        base = img.filter(ImageFilter.UnsharpMask(radius=2, percent=120, threshold=3))
        base = ImageEnhance.Contrast(base).enhance(1.08)
        return base

    # default: grade
    base = ImageEnhance.Color(img).enhance(1.05)
    base = ImageEnhance.Contrast(base).enhance(1.10)
    base = base.filter(ImageFilter.UnsharpMask(radius=2, percent=110, threshold=3))
    return base


def img2img_arrange(base: Image.Image, prompt: str, negative: str, seed: int) -> Image.Image:
    import torch
    from diffusers import AutoPipelineForImage2Image

    # choose device
    if DEVICE == "cuda":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif DEVICE == "cpu":
        device = "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # CPU: float32 is safest. CUDA: fp16 is fastest.
    dtype = torch.float16 if device == "cuda" else torch.float32

    if TORCH_NUM_THREADS > 0:
        try:
            torch.set_num_threads(TORCH_NUM_THREADS)
        except Exception:
            pass
 
    # Prefer safetensors; try fp16 variant on CUDA when available
    pipe = None

    if device == "cuda":
        try:
            pipe = _from_pretrained_img2img(
                AutoPipelineForImage2Image,
                MODEL_ID,
                dtype,
                use_safetensors=True,
                variant="fp16",
            )
        except Exception:
            pipe = None
    if pipe is None:
        pipe = _from_pretrained_img2img(
            AutoPipelineForImage2Image,
            MODEL_ID,
            dtype,
            use_safetensors=True,
        )
    pipe = pipe.to(device)

    try:
        pipe.set_progress_bar_config(disable=True)
    except Exception:
        pass
    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass
    # diffusers newer API prefers pipe.vae.enable_slicing/tiling
    try:
        pipe.vae.enable_slicing()
    except Exception:
        try:
            pipe.enable_vae_slicing()
        except Exception:
            pass
    try:
        pipe.vae.enable_tiling()
    except Exception:
        try:
            pipe.enable_vae_tiling()
        except Exception:
            pass

    g = torch.Generator(device=device).manual_seed(seed)


    # Avoid CLIP max token warnings (77 tokens)
    prompt = _clip_safe_prompt(pipe, prompt)
    negative = _clip_safe_prompt(pipe, negative)


    # Mirror-augment:
    # - mirror input (L/R flip) before img2img
    # - generate
    # - mirror output back (so final orientation matches original)
    # NOTE: flipping twice without generation is identical; the "difference" comes from the img2img step in between.
    in_img = ImageOps.mirror(base) if MIRROR_AUG else base

    with torch.inference_mode():
        out_img = pipe(
            prompt=prompt,
            negative_prompt=negative,
            image=in_img,
            strength=max(0.05, min(0.95, STRENGTH)),
            guidance_scale=GUIDANCE_SCALE,
            num_inference_steps=max(1, min(4, STEPS)),
            generator=g,
        ).images[0].convert("RGB")
    out = ImageOps.mirror(out_img) if MIRROR_AUG else out_img

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
            it["permalink"] = f"./?post={feed_stem}"
            it["image"] = rel_url
            it["image_url"] = rel_url
            it["image_prompt"] = prompt
            it["image_model"] = MODEL_ID
            it["image_mirror_aug"] = bool(MIRROR_AUG)
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
        # Optional: run SD at smaller size for CPU speed
        sdw = _floor_to_8(SD_W)
        sdh = _floor_to_8(SD_H)
        sd_in = base
        if sdw and sdh and (sdw != PAGE_W or sdh != PAGE_H):
            sd_in = fit_cover(base, sdw, sdh)

        arranged = img2img_arrange(sd_in, PROMPT_LOCAL, NEGATIVE_LOCAL, seed=seed)
        arranged = fit_cover(arranged, PAGE_W, PAGE_H)
    else:
        arranged = pillow_arrange(base, STYLE)

    arranged.save(out_path, format="PNG", optimize=True)

    # JSON PATCH_JSON=0
    if PATCH_JSON:
        now_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

        rel_url = ""
        try:
            rel_url = "./" + str(out_path.relative_to(Path("frontend/app/public"))).replace("\\", "/")
        except Exception:
            # OUT_DIR
            rel_url = ""

        # Update latest.json (keep existing keys)
        if isinstance(latest, dict):
            latest["image"] = rel_url
            latest["image_url"] = rel_url
            latest["image_prompt"] = PROMPT_LOCAL if MODE == "img2img" else f"pillow:{STYLE}"
            latest["id"] = feed_stem
            latest["permalink"] = f"./?post={feed_stem}"
            latest["image_model"] = MODEL_ID if MODE == "img2img" else "pillow"
            latest["image_mirror_aug"] = bool(MIRROR_AUG)
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
