
import os, re, json, hashlib
from pathlib import Path
from datetime import datetime, timezone

import torch
from diffusers import DiffusionPipeline
from PIL import Image, ImageDraw, ImageFont

LATEST_PATH = Path(os.environ["LATEST_PATH"])
FEED_DIR = Path(os.environ.get("FEED_DIR", "frontend/app/public/feed"))
OUT_DIR = Path(os.environ.get("OUT_DIR", "frontend/app/public/image"))
MODEL_ID = os.environ.get("MODEL_ID", "stabilityai/sd-turbo")
STEPS = int(os.environ.get("STEPS", "2"))
BASE_SIZE = int(os.environ.get("BASE_SIZE"))
PAGE_W = int(os.environ.get("PAGE_W"))
PAGE_H = int(os.environ.get("PAGE_H"))
BRAND = os.environ.get("BRAND", "GOODDAY YOKOSUKA")
PLACE = os.environ.get("PLACE", "")

def load_json(p: Path):
  return json.loads(p.read_text(encoding="utf-8"))

def dump_json(p: Path, obj):
  p.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

def pick_font(size: int, bold: bool = False):
  candidates = []
  if bold:
    candidates += [
      "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
      "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
  else:
    candidates += [
      "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
      "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf",
    ]
  for p in candidates:
    try:
      return ImageFont.truetype(p, size=size)
    except Exception:
      pass
  return ImageFont.load_default()

def clean_for_prompt(text: str) -> str:
  t = re.sub(r"https?://\S+", "", text)
  t = re.sub(r"#\w+", "", t)
  t = re.sub(r"\s+", " ", t).strip()
  return t[:220]

def slug(s: str) -> str:
  s = re.sub(r"\s+", "_", s.strip())
  s = re.sub(r"[^0-9A-Za-z_\-]", "", s)
  return s[:60] or "item"

latest = load_json(LATEST_PATH)
if not isinstance(latest, dict) or not latest.get("date") or not latest.get("text"):
  print("latest.json not in expected single-object shape (date/text). Skip.")
  raise SystemExit(0)

date = str(latest.get("date", "")).strip()
text = str(latest.get("text", "")).strip()
place = str(latest.get("place", "") or PLACE).strip()
generated_at = str(latest.get("generated_at", "")).strip()

# Deterministic 8-hex seed derived from the feed content.
# Used for the Stable Diffusion seed and as a fallback stem when no feed file is found.
seed_src = f"{date}\n{place}\n{generated_at}\n{text}".encode("utf-8")
seed_hex8 = hashlib.sha1(seed_src).hexdigest()[:8]

feeds = []
if FEED_DIR.exists():
    feeds = sorted(FEED_DIR.glob("feed_*.json"), key=lambda x: x.name, reverse=True)

feed_stem = feeds[0].stem if feeds else f"{slug(date)}_{seed_hex8}"
fn = f"{feed_stem}.png"
OUT_DIR.mkdir(parents=True, exist_ok=True)
out_path = OUT_DIR / fn
rel_url = "/" + str(out_path.relative_to(Path("frontend/app/public"))).replace("\\", "/")

if latest.get("image_url") == rel_url and out_path.exists():
  print("Already generated; skip.")
  raise SystemExit(0)

core = clean_for_prompt(text)
prompt = (
  "Japanese picture illustration, illustrated image style, "
  "watercolor and colored pencil, soft warm light, cute and simple, "
  "hand-drawn, minimal details, friendly atmosphere, "
  f"scene in {place}. Inspired by this message: {core}"
)
negative = (
  "nsfw, nude, sexual, gore, violence, hate, disturbing, "
  "text, watermark, logo, letters, words, typography, caption, subtitles, signature, "
  "low quality, blurry"
)

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32
pipe = DiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype).to(device)

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

guidance_scale = 0.0
seed = int(seed_hex8, 16)
g = torch.Generator(device=device).manual_seed(seed)

img = pipe(
  prompt=prompt,
  negative_prompt=negative,
  guidance_scale=guidance_scale,
  num_inference_steps=max(1, min(4, STEPS)),
  width=PAGE_W,
  height=PAGE_H,
  generator=g,
).images[0].convert("RGB")

page = img.convert("RGB")
page.save(out_path, format="PNG", optimize=True)

now_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
latest["image_url"] = rel_url
latest["image_prompt"] = prompt
latest["image_model"] = MODEL_ID
latest["image_generated_at"] = now_iso
dump_json(LATEST_PATH, latest)

def patch_feed_file(p: Path) -> bool:
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

if FEED_DIR.exists():
  feeds = sorted(FEED_DIR.glob("feed_*.json"), key=lambda x: x.name, reverse=True)
  if feeds:
    patch_feed_file(feeds[0])

print("Generated:", out_path, "url:", rel_url)