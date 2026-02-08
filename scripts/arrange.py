import os, re, json, hashlib
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

from PIL import Image, ImageFilter, ImageEnhance, ImageOps, ImageFile

# Network snapshots can occasionally be partially downloaded.
# Be tolerant here, while we also harden the downloader.
ImageFile.LOAD_TRUNCATED_IMAGES = True

LATEST_PATH = Path(os.environ.get("LATEST_PATH", "frontend/app/public/latest.json"))
FEED_DIR = Path(os.environ.get("FEED_DIR", "frontend/app/public/feed"))
OUT_DIR = Path(os.environ.get("OUT_DIR", "frontend/app/public/image"))

PAGE_W = int(os.environ.get("PAGE_W", "640"))
PAGE_H = int(os.environ.get("PAGE_H", "480"))

INPUT_IMAGE = Path(os.environ.get("INPUT_IMAGE", "snapshot.jpg"))
MODE = os.environ.get("MODE").strip()          # img2img
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

MODEL_ID = os.environ.get("MODEL_ID", "stabilityai/sdxl-turbo").strip()
LORA_PATH = os.environ.get("LORA_PATH", "").strip()
LORA_SCALE = float(os.environ.get("LORA_SCALE", "0.8"))
PLACE = os.environ.get("PLACE", "").strip()

PROMPT = os.environ.get("PROMPT").strip()
NEGATIVE = os.environ.get("NEGATIVE").strip()

# NOTE:
MIRROR_AUG = (os.environ.get("MIRROR_AUG", "0").strip() != "0")

PATCH_JSON = (os.environ.get("PATCH_JSON", "1").strip() != "0")
OUT_NAME = os.environ.get("OUT_NAME", "").strip()

def _sanitize_img2img_params(steps: int, strength: float) -> tuple[int, float]:
    s = max(1, int(steps))
    st = float(strength)
    st = max(0.05, min(0.95, st))
    min_st = (1.0 / s) + 1e-3
    if st < min_st:
        st = min_st
    return s, st

def load_json(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def dump_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def clean_for_prompt(text: str) -> str:
    t = re.sub(r"https?://\S+", "", text or "")
    t = re.sub(r"\s+", " ", t).strip()
    return t[:220]


def slug(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s or "x"


def fit_cover(img: Image.Image, w: int, h: int) -> Image.Image:
    iw, ih = img.size
    if iw <= 0 or ih <= 0:
        raise ValueError(f"invalid image size: {img.size}")
    scale = max(w / iw, h / ih)
    nw = max(1, int(round(iw * scale)))
    nh = max(1, int(round(ih * scale)))
    resized = img.resize((nw, nh), Image.BICUBIC)
    left = max(0, (nw - w) // 2)
    top = max(0, (nh - h) // 2)
    return resized.crop((left, top, left + w, top + h))


def pillow_arrange(img: Image.Image, style: str) -> Image.Image:
    style = (style or "grade").strip().lower()
    out = img

    if style == "grade":
        out = ImageEnhance.Contrast(out).enhance(1.15)
        out = ImageEnhance.Color(out).enhance(1.12)
        out = ImageEnhance.Sharpness(out).enhance(1.05)
    elif style == "oil":
        out = out.filter(ImageFilter.SMOOTH_MORE)
        out = ImageEnhance.Color(out).enhance(1.18)
        out = ImageEnhance.Contrast(out).enhance(1.08)
    elif style == "watercolor":
        out = out.filter(ImageFilter.SMOOTH_MORE)
        out = ImageEnhance.Color(out).enhance(0.95)
        out = ImageEnhance.Contrast(out).enhance(1.06)
    elif style == "photo":
        out = ImageEnhance.Contrast(out).enhance(1.05)
        out = ImageEnhance.Sharpness(out).enhance(1.10)
    return out


def _floor_to_8(x: int) -> int:
    if x <= 0:
        return 0
    return max(8, (x // 8) * 8)


def _pick_device():
    import torch

    if DEVICE == "cpu":
        return torch.device("cpu")
    if DEVICE == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _clip_safe_prompt(pipe, text: str, max_tokens: int = 75) -> str:
    text = (text or "").strip()
    if not text:
        return "photorealistic photo"

    tok = getattr(pipe, "tokenizer", None)
    if tok is None:
        return text

    try:
        enc = tok(text, truncation=False, add_special_tokens=True)
        ids = enc["input_ids"]
        if len(ids) <= 77:
            return text
        ids2 = ids[:max_tokens]
        return tok.decode(ids2, skip_special_tokens=True).strip() or text[:200]
    except Exception:
        return text[:200]


def _from_pretrained_img2img(model_id: str, *, torch_dtype, **kwargs):
    from diffusers import AutoPipelineForImage2Image

    try:
        return AutoPipelineForImage2Image.from_pretrained(model_id, torch_dtype=torch_dtype, **kwargs)
    except TypeError as e:
        if "torch_dtype" in str(e):
            return AutoPipelineForImage2Image.from_pretrained(model_id, dtype=torch_dtype, **kwargs)
        raise


def img2img_arrange(base: Image.Image, prompt: str, negative: str, *, seed: int) -> Image.Image:
    import torch

    device = _pick_device()
    if TORCH_NUM_THREADS > 0 and device.type == "cpu":
        torch.set_num_threads(TORCH_NUM_THREADS)

    dtype = torch.float16 if device.type == "cuda" else torch.float32

    pipe = _from_pretrained_img2img(MODEL_ID, torch_dtype=dtype)

    try:
        pipe = pipe.to(device=device, dtype=dtype)
    except TypeError:
        pipe = pipe.to(device)
        try:
            pipe = pipe.to(dtype=dtype)
        except TypeError:
            pass

    if device.type == "cpu":
        try:
            if getattr(pipe, "dtype", None) == torch.float16:
                pipe = pipe.to(dtype=torch.float32)
        except Exception:
            pass

    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=True)
    if hasattr(pipe, "enable_attention_slicing"):
        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass
    if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_slicing"):
        try:
            pipe.vae.enable_slicing()
        except Exception:
            pass

    lora_tag = ""
    if LORA_PATH:
        p = Path(LORA_PATH)
        if not p.exists():
            raise FileNotFoundError(f"LORA_PATH not found: {p}")
        pipe.load_lora_weights(str(p))
        lora_tag = p.name

    prompt2 = _clip_safe_prompt(pipe, prompt)
    negative2 = _clip_safe_prompt(pipe, negative) if (negative or "").strip() else ""
    prompt2 = (prompt2 or "").strip() or "photorealistic photo"

    g = torch.Generator(device=device).manual_seed(int(seed))

    img_in = base
    if MIRROR_AUG:
        img_in = ImageOps.mirror(img_in)

    steps_i, strength_f = _sanitize_img2img_params(STEPS, STRENGTH)
    with torch.inference_mode():
        out = pipe(
            prompt=prompt2,
            negative_prompt=negative2,
            image=img_in,
            strength=strength_f,
            guidance_scale=float(GUIDANCE_SCALE),
            num_inference_steps=steps_i,
            generator=g,
            **({"cross_attention_kwargs": {"scale": float(LORA_SCALE)}} if LORA_PATH else {}),
        )

    out_img = out.images[0]
    if MIRROR_AUG:
        out_img = ImageOps.mirror(out_img)
    return out_img


def _safe_str(x: Any) -> str:
    return str(x) if x is not None else ""


def _match_item(item: Any, *, date: str, text: str, generated_at: str) -> bool:
    if not isinstance(item, dict):
        return False
    same_dt = _safe_str(item.get("date")).strip() == date and _safe_str(item.get("text")).strip() == text
    same_ga = bool(generated_at) and _safe_str(item.get("generated_at")).strip() == generated_at
    return same_dt or same_ga


def patch_feed_file(
    feed_path: Path,
    *,
    date: str,
    text: str,
    generated_at: str,
    feed_stem: str,
    rel_url: str,
    prompt: str,
    now_iso: str,
) -> bool:
    if not feed_path.exists():
        return False

    try:
        obj = load_json(feed_path)
    except Exception:
        return False

    changed = False

    def patch_dict(d: dict) -> None:
        nonlocal changed
        d["image"] = rel_url
        d["image_url"] = rel_url
        d["image_prompt"] = prompt
        d["image_model"] = MODEL_ID
        if MODE == "img2img" and LORA_PATH:
            d["image_lora"] = Path(LORA_PATH).name
            d["image_lora_scale"] = float(LORA_SCALE)
        d["image_mirror_aug"] = bool(MIRROR_AUG)
        d["image_generated_at"] = now_iso
        d["id"] = feed_stem
        d["permalink"] = f"./?post={feed_stem}"
        changed = True

    if isinstance(obj, dict) and isinstance(obj.get("items"), list):
        for it in obj["items"]:
            if _match_item(it, date=date, text=text, generated_at=generated_at):
                patch_dict(it)
                break
    elif isinstance(obj, list):
        for it in obj:
            if _match_item(it, date=date, text=text, generated_at=generated_at):
                patch_dict(it)
                break
    elif isinstance(obj, dict):
        # snapshot single-object
        if _match_item(obj, date=date, text=text, generated_at=generated_at) or True:
            patch_dict(obj)

    if changed:
        dump_json(feed_path, obj)
    return changed


def main() -> int:
    latest = {}
    if LATEST_PATH.exists():
        try:
            latest = load_json(LATEST_PATH)
        except Exception:
            latest = {}

    feeds = []
    if FEED_DIR.exists():
        feeds = sorted(FEED_DIR.glob("feed_*.json"), key=lambda x: x.name, reverse=True)

    date = _safe_str(getattr(latest, "get", lambda *_: "")("date")).strip() if isinstance(latest, dict) else ""
    text = _safe_str(getattr(latest, "get", lambda *_: "")("text")).strip() if isinstance(latest, dict) else ""
    generated_at = _safe_str(getattr(latest, "get", lambda *_: "")("generated_at")).strip() if isinstance(latest, dict) else ""
    place = (PLACE or (_safe_str(latest.get("place")) if isinstance(latest, dict) else "")).strip()

    if (not date or not text) and feeds:
        try:
            snap = load_json(feeds[0])
            if isinstance(snap, dict):
                date = date or _safe_str(snap.get("date")).strip()
                text = text or _safe_str(snap.get("text")).strip()
                generated_at = generated_at or _safe_str(snap.get("generated_at")).strip()
                if not place:
                    place = _safe_str(snap.get("place")).strip()
        except Exception:
            pass

    seed_src = f"{date}\n{place}\n{generated_at}\n{text}".encode("utf-8")
    seed_hex8 = hashlib.sha1(seed_src).hexdigest()[:8]
    seed = int(seed_hex8, 16)

    if SEED_OVERRIDE:
        try:
            seed = int(SEED_OVERRIDE)
        except Exception:
            pass
    seed += int(SEED_OFFSET)

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
        prompt_local = (
            "photorealistic photo, natural colors, documentary style, "
            f"scene in {place or 'Yokosuka, Japan'}, "
            "same scene and composition as the input image. "
            f"Inspired by: {core}"
        )
    else:
        prompt_local = PROMPT

    if not NEGATIVE:
        negative_local = (
            "anime, illustration, cartoon, CGI, fantasy, unreal, "
            "text, watermark, logo, letters, words, typography, caption, subtitles, signature, "
            "low quality, blurry"
        )
    else:
        negative_local = NEGATIVE

    if MODE == "img2img":
        sdw = _floor_to_8(SD_W)
        sdh = _floor_to_8(SD_H)
        sd_in = base
        if sdw and sdh and (sdw != PAGE_W or sdh != PAGE_H):
            sd_in = fit_cover(base, sdw, sdh)

        arranged = img2img_arrange(sd_in, prompt_local, negative_local, seed=seed)
        arranged = fit_cover(arranged, PAGE_W, PAGE_H)
    else:
        arranged = pillow_arrange(base, STYLE)

    arranged.save(out_path, format="PNG", optimize=True)

    if PATCH_JSON and isinstance(latest, dict):
        now_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        rel_url = ""
        try:
            rel_url = "./" + str(out_path.relative_to(Path("frontend/app/public"))).replace("\\", "/")
        except Exception:
            rel_url = ""

        latest["image"] = rel_url
        latest["image_url"] = rel_url
        latest["image_prompt"] = prompt_local if MODE == "img2img" else f"pillow:{STYLE}"
        latest["id"] = feed_stem
        latest["permalink"] = f"./?post={feed_stem}"
        latest["image_model"] = MODEL_ID
        if MODE == "img2img" and LORA_PATH:
            latest["image_lora"] = Path(LORA_PATH).name
            latest["image_lora_scale"] = float(LORA_SCALE)
        latest["image_mirror_aug"] = bool(MIRROR_AUG)
        latest["image_generated_at"] = now_iso
        dump_json(LATEST_PATH, latest)

        if feeds:
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
