import os, time, requests
url = os.environ["SNAP_URL"].strip()
sep = "&" if "?" in url else "?"
url2 = f"{url}{sep}t={int(time.time())}"
headers = {
  "User-Agent": "snapshot-safe-http/1.0 (+github-actions)",
  "Accept": "image/*",
  "Cache-Control": "no-cache",
  "Pragma": "no-cache",
}
r = requests.get(url2, headers=headers, timeout=30)
r.raise_for_status()
ctype = (r.headers.get("Content-Type","") or "").lower()
if "image" not in ctype:
  raise RuntimeError(f"Non-image response: Content-Type={ctype}")
with open("snapshot.jpg","wb") as f:
  f.write(r.content)
print("OK snapshot.jpg", len(r.content), ctype)