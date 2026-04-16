import React, { useCallback, useEffect, useMemo, useState,useRef } from "react";
import {
  ActivityIndicator,
  Animated,
  FlatList,
  Image,
  Linking,
  Platform,
  RefreshControl,
  Text,
  useWindowDimensions,
  View,
  Pressable,
} from "react-native";

const RAW_CONTACT_URL = (process.env.EXPO_PUBLIC_FEEDBACK_FORM_URL ?? "").trim();
const CONTACT_FORM_URL =
  RAW_CONTACT_URL.startsWith("http://") || RAW_CONTACT_URL.startsWith("https://")
    ? RAW_CONTACT_URL
    : "";
const RAW_ADS_URL = (process.env.EXPO_PUBLIC_ADS_FORM_URL ?? "").trim();
const ADS_FORM_URL =
  RAW_ADS_URL.startsWith("http://") || RAW_ADS_URL.startsWith("https://")
    ? RAW_ADS_URL
    : "";

type FeedLink = {
  title: string;
  url: string;
};

type FeedItem = {
  id: string;
  date: string; // YYYY-MM-DD
  text: string;
  place?: string;
  kind?: string;
  avatar_image?: string;
  generated_at?: string; // ISO string (often Z)
  image?: string; // local path or absolute URL
  image_prompt?: string; // optional (for matching)
  links?: FeedLink[];
};

type Feed = {
  updated_at?: string;
  place?: string;
  items: FeedItem[];
};

type GuideItem = {
  id: string;
  slug?: string;
  title: string;
  summary?: string;
  description?: string;
  place?: string;
  published_at?: string;
  date?: string;
  hero_image?: string;
  permalink?: string;
};

type GuidesIndex = {
  updated_at?: string;
  items: GuideItem[];
};

type SlotItem = {
  kind: "ad";
  id: string;
  title: string;
  body: string;
  cta?: string;
  url?: string;
  sponsor?: string;
  disclaimer?: string;
  emoji?: string;
};

type TimelineItem = FeedItem | SlotItem;

function normalizeLinks(raw: any): FeedLink[] | undefined {
  const items = Array.isArray(raw) ? raw : raw == null ? [] : [raw];
  const out: FeedLink[] = [];
  const seen = new Set<string>();

  for (const item of items) {
    let title = "";
    let url = "";

    if (typeof item === "string") {
      url = item.trim();
      title = url;
    } else if (item && typeof item === "object") {
      url = typeof item.url === "string" ? item.url.trim() : "";
      title = typeof item.title === "string" ? item.title.trim() : url;
    }

    if (!url || seen.has(url)) continue;
    seen.add(url);
    out.push({ title: title || url, url });
  }

  return out.length ? out : undefined;
}


function isSlotItem(it: TimelineItem): it is SlotItem {
  return (it as any)?.kind === "ad";
}

const APP_BG = "#f6f4ff";
const CARD_BG = "#ffffff";
const TEXT_DIM = "#333333";

const BORDER = "#000000";
const BUBBLE_RADIUS = 16;
const BUBBLE_BORDER_W = 2;

const CONTENT_MAX_W = 760;
const MASCOT_COL_W = 128;
const MASCOT_SIZE = 96;
const MASCOT_RADIUS = 18; // rounded rectangle for avatar frame
const MASCOT_BORDER_W = 2;
const DEFAULT_AVATAR_IMAGE = "image/avatar/normal.png";
const SIDEBAR_W = 240;

const FEED_SCROLL_ID = "feed-scroll";
const MAX_DEEP_LINK_ATTEMPTS = 25;

const ITEM_EVERY_N = Math.max(2, Number((process.env.EXPO_PUBLIC_ITEM_EVERY_N || "5").trim()) || 5); // 1 ad per N items
const ITEM_BG = "#fff7ed";
const ITEM_BADGE_BG = "#fb923c";

const FAKE_ITEM_TEMPLATES: Omit<SlotItem, "id" | "kind">[] = [
  {
    title: "demo1",
    body: "demo1",
    cta: "check",
    url: ADS_FORM_URL,
    sponsor: "demo1",
    disclaimer: "demo1",
    emoji: "🧜‍♀️",
  },
  {
    title: "demo1",
    body: "demo1",
    cta: "check",
    url: ADS_FORM_URL,
    sponsor: "demo1",
    disclaimer: "demo1",
    emoji: "🧜‍♀️",
  },
  {
    title: "demo1",
    body: "demo1",
    cta: "check",
    url: ADS_FORM_URL,
    sponsor: "demo1",
    disclaimer: "demo1",
    emoji: "🧜‍♀️",
  },
  {
    title: "demo1",
    body: "demo1",
    cta: "check",
    url: ADS_FORM_URL,
    sponsor: "demo1",
    disclaimer: "demo1",
    emoji: "🧜‍♀️",
  },
  {
    title: "demo1",
    body: "demo1",
    cta: "check",
    url: ADS_FORM_URL,
    sponsor: "demo1",
    disclaimer: "demo1",
    emoji: "🧜‍♀️",
  },
];

type SlotBanner = {
  id: string;
  title: string;
  body: string;
  cta: string;
  url: string;
  imageUri: string;
  sponsor?: string;
  disclaimer?: string;
};

type GuideBanner = {
  id: string;
  kind: "guide" | "ad";
  title: string;
  summary: string;
  url: string;
  imageUri: string;
  meta?: string;
  badgeLabel?: string;
  headerLabel?: string;
  ctaLabel?: string;
  disclaimer?: string;
};

const GUIDE_RECRUIT_EVERY_N = Math.max(
  4,
  Number((process.env.EXPO_PUBLIC_GUIDE_RECRUIT_EVERY_N || "4").trim()) || 4
);

const SLOT_ROTATE_MS = Math.max(2500, Number((process.env.EXPO_PUBLIC_SLOT_ROTATE_MS || "6500").trim()) || 6500);
const SLOT_FADE_MS = Math.max(200, Number((process.env.EXPO_PUBLIC_SLOT_FADE_MS || "800").trim()) || 800);

const SLOT_BANNERS: SlotBanner[] = [
  {
    id: "slot-0",
    title: "Ocean view, zero effort",
    body: "",
    cta: "Open demo",
    url: ADS_FORM_URL,
    imageUri: "https://picsum.photos/seed/goodday_ocean/900/650",
    sponsor: "GOODDAY",
    disclaimer: "Demo ad slot — not a real promotion.",
  },
  {
    id: "slot-1",
    title: "Coffee & quiet time",
    body: "",
    cta: "See more",
    url: ADS_FORM_URL,
    imageUri: "https://picsum.photos/seed/goodday_coffee/900/650",
    sponsor: "GOODDAY",
    disclaimer: "Demo ad slot — not a real promotion.",
  },
  {
    id: "slot-2",
    title: "Weekend micro trip",
    body: "",
    cta: "View route",
    url: ADS_FORM_URL,
    imageUri: "https://picsum.photos/seed/goodday_trip/900/650",
    sponsor: "GOODDAY",
    disclaimer: "Demo ad slot — not a real promotion.",
  },
  {
    id: "slot-3",
    title: "Sunset soundtrack",
    body: "",
    cta: "Play",
    url: ADS_FORM_URL,
    imageUri: "https://picsum.photos/seed/goodday_sunset/900/650",
    sponsor: "GOODDAY",
    disclaimer: "Demo ad slot — not a real promotion.",
  },
  {
    id: "slot-4",
    title: "Mountain air",
    body: "",
    cta: "Learn more",
    url: ADS_FORM_URL,
    imageUri: "https://picsum.photos/seed/goodday_mountain/900/650",
    sponsor: "GOODDAY",
    disclaimer: "Demo ad slot — not a real promotion.",
  },
  {
    id: "slot-5",
    title: "City lights",
    body: "",
    cta: "Open",
    url: ADS_FORM_URL,
    imageUri: "https://picsum.photos/seed/goodday_city/900/650",
    sponsor: "GOODDAY",
    disclaimer: "Demo ad slot — not a real promotion.",
  },
];

function hashString(s: string): number {
  // Simple deterministic hash (for stable rotation per anchor id)
  let h = 0;
  for (let i = 0; i < s.length; i += 1) h = (h * 31 + s.charCodeAt(i)) | 0;
  return Math.abs(h);
}

function makeAnchor(anchorId: string): SlotItem {
  const idx = FAKE_ITEM_TEMPLATES.length ? hashString(anchorId) % FAKE_ITEM_TEMPLATES.length : 0;
  const t = FAKE_ITEM_TEMPLATES[idx] ?? FAKE_ITEM_TEMPLATES[0];
  return {
    kind: "ad",
    id: `ad|${anchorId}`,
    title: t?.title ?? "Sponsored",
    body: t?.body ?? "Demo ad",
    cta: t?.cta,
    url: t?.url,
    sponsor: t?.sponsor,
    disclaimer: t?.disclaimer,
    emoji: t?.emoji,
  };
}

function interleaveAds(posts: FeedItem[]): TimelineItem[] {
  // every Nth item is an ad (i.e. after N-1 posts)
  const n = ITEM_EVERY_N;
  const afterPosts = Math.max(1, n - 1);

  const out: TimelineItem[] = [];
  let count = 0;

  for (const p of posts) {
    out.push(p);
    count += 1;

    if (count % afterPosts === 0) {
      out.push(makeAnchor(p.id));
    }
  }

  return out;
}

function ensureWebScrollbarStyle() {
  if (Platform.OS !== "web") return;

  const STYLE_ID = "hide-scrollbar-style";
  if (document.getElementById(STYLE_ID)) return;

  const style = document.createElement("style");
  style.id = STYLE_ID;
  style.textContent = `
    /* target only the FlatList scroll node */
    #${FEED_SCROLL_ID} {
      -ms-overflow-style: none;   /* IE/Edge legacy */
      scrollbar-width: none;      /* Firefox */
    }
    #${FEED_SCROLL_ID}::-webkit-scrollbar {
      width: 0px;
      height: 0px;
      display: none;              /* Chrome/Safari */
    }
  `;
  document.head.appendChild(style);
}

function parseTimeLike(input: string): Date | null {
  const s = String(input ?? "").trim();
  if (!s) return null;

  if (/(Z|[+-]\d{2}:\d{2})$/.test(s)) {
    const d = new Date(s);
    return Number.isNaN(d.getTime()) ? null : d;
  }

  if (/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(:\d{2})?$/.test(s)) {
    const d = new Date(`${s}+09:00`);
    return Number.isNaN(d.getTime()) ? null : d;
  }

  const d = new Date(s);
  return Number.isNaN(d.getTime()) ? null : d;
}

function formatJst(isoLike: string, withSeconds = false): string {
  const d = parseTimeLike(isoLike);
  if (!d) return isoLike;

  const jstMs = d.getTime() + 9 * 60 * 60 * 1000;
  const j = new Date(jstMs);
  const pad = (n: number) => String(n).padStart(2, "0");

  const yyyy = j.getUTCFullYear();
  const mm = pad(j.getUTCMonth() + 1);
  const dd = pad(j.getUTCDate());
  const hh = pad(j.getUTCHours());
  const mi = pad(j.getUTCMinutes());
  const ss = pad(j.getUTCSeconds());
  return `${yyyy}-${mm}-${dd} ${hh}:${mi}${withSeconds ? `:${ss}` : ""} JST`;
}

function safeJsonParse(raw: string): unknown | null {
  try {
    return JSON.parse(raw) as unknown;
  } catch {
    return null;
  }
}

function normalizeFeed(parsed: unknown): Feed | null {
  if (!parsed) return null;

  if (typeof parsed === "object" && !Array.isArray(parsed)) {
    const obj = parsed as any;

    if (Array.isArray(obj.items)) {
      const items: FeedItem[] = obj.items
        .map((it: any, idx: number): FeedItem | null => {
          const date = typeof it?.date === "string" ? it.date : "";
          const text = typeof it?.text === "string" ? it.text : "";
          if (!date || !text) return null;
          const id = typeof it?.id === "string" ? it.id : `${date}-${idx}`;
          const place = typeof it?.place === "string" ? it.place : undefined;
          const generated_at = typeof it?.generated_at === "string" ? it.generated_at : undefined;
          const image =
            typeof it?.image === "string"
              ? it.image
              : typeof it?.image_url === "string"
              ? it.image_url
              : typeof it?.imageUri === "string"
              ? it.imageUri
              : undefined;
          const image_prompt = typeof it?.image_prompt === "string" ? it.image_prompt : undefined;
          const kind = typeof it?.kind === "string" ? it.kind : undefined;
          const avatar_image =
            typeof it?.avatar_image === "string"
              ? it.avatar_image
              : typeof it?.avatarImage === "string"
                ? it.avatarImage
                : typeof it?.avatar === "string"
                  ? it.avatar
                  : typeof it?.avatar_url === "string"
                    ? it.avatar_url
                    : undefined;
          const links = normalizeLinks(it?.links ?? it?.link);
          return { 
            id, 
            date, 
            text, 
            place, 
            generated_at, 
            image, 
            image_prompt, 
            kind,
            avatar_image,
            links 
          };
        })
        .filter(Boolean) as FeedItem[];

      return {
        updated_at: typeof obj.updated_at === "string" ? obj.updated_at : undefined,
        place: typeof obj.place === "string" ? obj.place : undefined,
        items,
      };
    }

    const date = typeof obj.date === "string" ? obj.date : "";
    const text = typeof obj.text === "string" ? obj.text : "";
    if (date && text) {
      const id = typeof obj.id === "string" ? obj.id : `${date}-0`;
      const place = typeof obj.place === "string" ? obj.place : undefined;
      const generated_at = typeof obj.generated_at === "string" ? obj.generated_at : undefined;
      const image =
        typeof obj?.image === "string"
          ? obj.image
          : typeof obj?.image_url === "string"
          ? obj.image_url
          : typeof obj?.imageUri === "string"
          ? obj.imageUri
          : undefined;
      const image_prompt = typeof obj?.image_prompt === "string" ? obj.image_prompt : undefined;
      const kind = typeof obj?.kind === "string" ? obj.kind : undefined;
      const avatar_image =
        typeof obj?.avatar_image === "string"
          ? obj.avatar_image
          : typeof obj?.avatarImage === "string"
            ? obj.avatarImage
            : typeof obj?.avatar === "string"
              ? obj.avatar
              : typeof obj?.avatar_url === "string"
                ? obj.avatar_url
                : undefined;
      const updated_at = generated_at;
      const links = normalizeLinks(obj?.links ?? obj?.link);
      return {
        updated_at,
        place,
        items: [{ id, date, text, place, generated_at, image, image_prompt, kind, avatar_image, links }],
      };
    }
  }

  if (Array.isArray(parsed)) {
    const items: FeedItem[] = parsed
      .map((it: any, idx: number): FeedItem | null => {
        const date = typeof it?.date === "string" ? it.date : "";
        const text = typeof it?.text === "string" ? it.text : "";
        if (!date || !text) return null;
        const id = typeof it?.id === "string" ? it.id : `${date}-${idx}`;
        const place = typeof it?.place === "string" ? it.place : undefined;
        const generated_at = typeof it?.generated_at === "string" ? it.generated_at : undefined;
          const image =
            typeof it?.image === "string"
              ? it.image
              : typeof it?.image_url === "string"
              ? it.image_url
              : typeof it?.imageUri === "string"
              ? it.imageUri
              : undefined;
          const image_prompt = typeof it?.image_prompt === "string" ? it.image_prompt : undefined;
          const kind = typeof it?.kind === "string" ? it.kind : undefined;
          const avatar_image =
            typeof it?.avatar_image === "string"
              ? it.avatar_image
              : typeof it?.avatarImage === "string"
                ? it.avatarImage
                : typeof it?.avatar === "string"
                  ? it.avatar
                  : typeof it?.avatar_url === "string"
                    ? it.avatar_url
                    : undefined;
          const links = normalizeLinks(it?.links ?? it?.link);
          return { id, date, text, place, generated_at, image, image_prompt, kind, avatar_image, links };
        })
      .filter(Boolean) as FeedItem[];

    const last = parsed.length > 0 ? (parsed[parsed.length - 1] as any) : null;
    const updated_at = typeof last?.generated_at === "string" ? last.generated_at : undefined;
    const place = typeof last?.place === "string" ? last.place : undefined;

    return { updated_at, place, items };
  }

  return null;
}

function normalizeGuidesIndex(parsed: unknown): GuidesIndex | null {
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return null;
  const obj = parsed as any;
  if (!Array.isArray(obj.items)) return null;

  const items: GuideItem[] = obj.items
    .map((it: any): GuideItem | null => {
      const title = typeof it?.title === "string" ? it.title.trim() : "";
      if (!title) return null;

      const slug = typeof it?.slug === "string" ? it.slug.trim() : undefined;
      const permalink = typeof it?.permalink === "string" ? it.permalink.trim() : undefined;
      const id =
        typeof it?.id === "string" && it.id.trim()
          ? it.id.trim()
          : slug || permalink || title;

      return {
        id,
        slug,
        title,
        description: typeof it?.description === "string" ? it.description : undefined,
        summary: typeof it?.summary === "string" ? it.summary : undefined,
        place: typeof it?.place === "string" ? it.place : undefined,
        published_at: typeof it?.published_at === "string" ? it.published_at : undefined,
        date: typeof it?.date === "string" ? it.date : undefined,
        hero_image: typeof it?.hero_image === "string" ? it.hero_image : undefined,
        permalink,
      };
    })
    .filter(Boolean) as GuideItem[];

  return {
    updated_at: typeof obj.updated_at === "string" ? obj.updated_at : undefined,
    items,
  };
}

type ShareSdItem = {
  date?: string;
  place?: string;
  image: string;
  prompt?: string;
};

type ShareSdIndex = {
  updated_at?: string;
  items: ShareSdItem[];
};

function normalizeWebAssetPath(p: string): string {
  let s = String(p ?? "").trim();
  if (!s) return "";
  if (/^(https?:)?\/\//i.test(s) || s.startsWith("data:")) return s;

  if (Platform.OS === "web" && typeof window !== "undefined") {
    const baseSeg = window.location.pathname.split("/").filter(Boolean)[0] || "";
    // If the path already includes the repo segment (e.g. "rag_chat_bot/..."),
    // strip it to avoid double-prefixing when resolving relative URLs.
    if (baseSeg) {
      if (s.startsWith(`/${baseSeg}/`)) s = `./${s.slice(baseSeg.length + 2)}`;
      else if (s.startsWith(`${baseSeg}/`)) s = `./${s.slice(baseSeg.length + 1)}`;
    }
    // Treat leading "/" as repo-relative on GitHub Pages.
    if (s.startsWith("/")) return `.${s}`;
  }

  return s;
}

function isAbsoluteAssetUri(u: string): boolean {
  const s = String(u ?? "").trim();
  return /^(https?:)?\/\/|^data:|^file:|^blob:/i.test(s);
}

function normalizePublicAssetRelPath(p: string): string {
  let s = String(p ?? "").trim();
  if (!s) return "";
  if (isAbsoluteAssetUri(s)) return s;

  s = s.replace(/^\.\//, "").replace(/^\/+/, "");
  return s;
}

function buildSharePrompt(text: string, place?: string): string {
  const t = String(text ?? "").replace(/\s+/g, " ").trim().slice(0, 240);
  const p = String(place ?? "").trim();
  return p
    ? `cinematic illustration, ${p}, based on this short story: ${t}`
    : `cinematic illustration, based on this short story: ${t}`;
}


const FeedBubbleImage: React.FC<{ uris?: string[] }> = ({ uris }) => {
  const [idx, setIdx] = useState(0);
  const [hidden, setHidden] = useState(false);

  const key = useMemo(() => (uris ?? []).join("|"), [uris]);

  useEffect(() => {
    setIdx(0);
    setHidden(false);
  }, [key]);

  const uri = (uris ?? [])[idx] ?? "";
  if (!uri || hidden) return null;

  return (
    <View
      style={{
        marginTop: 10,
        marginBottom: 0,
        borderRadius: 12,
        overflow: "hidden",
        borderWidth: 1,
        borderColor: BORDER,
        backgroundColor: "#ffffff",
      }}
    >
      <Image
        source={{ uri }}
        style={{ width: "100%", aspectRatio: 4 / 3 }}
        resizeMode="cover"
        accessibilityLabel="image"
        onError={() => {
          if (uris && idx + 1 < uris.length) setIdx(idx + 1);
          else setHidden(true);
        }}
      />
    </View>
  );
};


function normalizeShareSdIndex(parsed: unknown): ShareSdIndex | null {
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return null;
  const obj = parsed as any;

  if (!Array.isArray(obj.items)) return null;

  const items: ShareSdItem[] = obj.items
    .map((it: any): ShareSdItem | null => {
      const image = typeof it?.image === "string" ? it.image : "";
      if (!image) return null;

      const date = typeof it?.date === "string" ? it.date : undefined;
      const place = typeof it?.place === "string" ? it.place : undefined;
      const prompt = typeof it?.prompt === "string" ? it.prompt : undefined;

      return { image, date, place, prompt };
    })
    .filter(Boolean) as ShareSdItem[];

  return {
    updated_at: typeof obj.updated_at === "string" ? obj.updated_at : undefined,
    items,
  };
}


function getFeedPointer(parsed: unknown): string | null {
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return null;
  const obj = parsed as any;

  const cand =
    typeof obj.feed_url === "string"
      ? obj.feed_url
      : typeof obj.feed_file === "string"
        ? obj.feed_file
        : typeof obj.feed_path === "string"
          ? obj.feed_path
          : null;

  if (!cand) return null;
  const s = String(cand).trim();
  return s ? s : null;
}

function getNextPointer(parsed: unknown): string | null {
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return null;
  const obj = parsed as any;

  const cand =
    typeof obj.next_url === "string"
      ? obj.next_url
      : typeof obj.next === "string"
        ? obj.next
        : typeof obj.nextPage === "string"
          ? obj.nextPage
          : typeof obj.next_page === "string"
            ? obj.next_page
            : null;

  if (!cand) return null;
  const s = String(cand).trim();
  return s ? s : null;
}

function resolveUrl(maybeRelative: string, baseUrl: string): string {
  try {
    if (maybeRelative.startsWith("http://") || maybeRelative.startsWith("https://")) return maybeRelative;
    if (typeof window !== "undefined") return new URL(maybeRelative, baseUrl).toString();
  } catch {
    // ignore
  }
  return maybeRelative;
}

function openResolvedUrl(url: string) {
  const raw = String(url ?? "").trim();
  if (!raw) return;
  const base = typeof window !== "undefined" ? window.location.href : raw;
  const target = resolveUrl(raw, base);
  void Linking.openURL(target).catch(() => {});
}

function addCacheBuster(url: string): string {
  const sep = url.includes("?") ? "&" : "?";
  return `${url}${sep}v=${Date.now()}`;
}

function getMascotUriForItem(item: FeedItem, assetBase: string): string | undefined {
  const raw = (item.avatar_image ?? "").trim();
  if (!raw) return undefined;
  const p = normalizePublicAssetRelPath(raw);
  if (isAbsoluteAssetUri(p)) return p;
  return resolveUrl(normalizeWebAssetPath(p), assetBase);
}

function Mascot({
  size = MASCOT_SIZE,
  uri,
  assetBase,
}: {
  size?: number;
  uri?: string;
  assetBase?: string;
}) {
  const envUri = (process.env.EXPO_PUBLIC_MASCOT_URI || "").trim();

  // Candidate order: per-item uri -> env -> default(normal)
  const primaryUri = useMemo(() => {
    const u = String(uri ?? "").trim();
    if (u) return u;

    const e = String(envUri ?? "").trim();
    if (!e) return "";

    if (isAbsoluteAssetUri(e)) return e;
    if (assetBase) {
      const p = normalizePublicAssetRelPath(e);
      if (isAbsoluteAssetUri(p)) return p;
      return resolveUrl(normalizeWebAssetPath(p), assetBase);
    }
    return "";
  }, [uri, envUri, assetBase]);

  const defaultUri = useMemo(() => {
    if (!assetBase) return "";
    const p = normalizePublicAssetRelPath(DEFAULT_AVATAR_IMAGE);
    if (isAbsoluteAssetUri(p)) return p;
    return resolveUrl(normalizeWebAssetPath(p), assetBase);
  }, [assetBase]);

  // If primary fails, fall back to default "normal" image.
  const [useFallback, setUseFallback] = useState(false);
  useEffect(() => {
    setUseFallback(false);
  }, [primaryUri, defaultUri]);

  const displayUri = useMemo(() => {
    if (useFallback || !primaryUri) return defaultUri || primaryUri;
    return primaryUri;
  }, [useFallback, primaryUri, defaultUri]);

  const Frame = ({ children }: { children: React.ReactNode }) => (
    <View
      style={{
        width: size,
        height: size,
        borderRadius: MASCOT_RADIUS,
        borderWidth: MASCOT_BORDER_W,
        borderColor: BORDER,
        overflow: "hidden",
        backgroundColor: "#ffffff",
        alignItems: "center",
        justifyContent: "center",
      }}
      accessibilityLabel="Mascot"
    >
      {children}
    </View>
  );

  if (displayUri) {
    return (
      <Frame>
        <Image
          source={{ uri: displayUri }}
          style={{ width: size, height: size }}
          accessibilityLabel="Mascot"
          resizeMode="cover"
          onError={() => {
            if (!useFallback && primaryUri && defaultUri && primaryUri !== defaultUri) setUseFallback(true);
          }}
        />
      </Frame>
    );
  }

  // Last resort (should be rare): keep a minimal placeholder
  return (
    <Frame>
      <View style={{ flex: 1, alignItems: "center", justifyContent: "center", backgroundColor: "#111111" }}>
        <Text style={{ color: "#ffffff", fontWeight: "512", fontSize: Math.max(18, Math.floor(size * 0.35)) }}>R</Text>
      </View>
    </Frame>
  );
}

type SlotCardVariant = "sidebar" | "inline";

function SlotCard({
  banners,
  startIndex,
  sticky = false,
  variant = "sidebar",
}: {
  banners: SlotBanner[];
  startIndex: number;
  sticky?: boolean;
  variant?: SlotCardVariant;
}) {
  const len = Math.max(1, banners.length);
  const safeStart = ((startIndex % len) + len) % len;

  const [active, setActive] = useState(safeStart);
  const [next, setNext] = useState((safeStart + 1) % len);

  // Cross-fade progress: 0 → show "active", 1 → show "next"
  const progress = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    // If banners length changes (it shouldn't), clamp indices.
    if (active >= len) setActive(0);
    if (next >= len) setNext((active + 1) % len);
  }, [active, next, len]);

  useEffect(() => {
    if (len <= 1) return;

    let cancelled = false;

    const interval = setInterval(() => {
      const n = (active + 1) % len;
      setNext(n);

      progress.stopAnimation();
      progress.setValue(0);

      Animated.timing(progress, {
        toValue: 1,
        duration: SLOT_FADE_MS,
        useNativeDriver: Platform.OS !== "web",
      }).start(({ finished }) => {
        if (!finished || cancelled) return;
        setActive(n);
        // Snap back to the stable state (active fully visible).
        progress.setValue(0);
      });
    }, SLOT_ROTATE_MS);

    return () => {
      cancelled = true;
      clearInterval(interval);
      progress.stopAnimation();
    };
  }, [active, len, progress]);

  const activeOpacity =
    len <= 1
      ? 1
      : progress.interpolate({
          inputRange: [0, 1],
          outputRange: [1, 0],
        });

  const nextOpacity = len <= 1 ? 0 : progress;

  const activeBanner = banners[active] ?? banners[0];
  const nextBanner = banners[next] ?? banners[0];

  const onPress = useCallback(() => {
    const url = activeBanner?.url;
    if (!url) return;
    void Linking.openURL(url).catch(() => {
      // ignore
    });
  }, [activeBanner?.url]);

  const imageAreaStyle =
    variant === "sidebar"
      ? ({ flex: 1, minHeight: 0, backgroundColor: "#e5e7eb" } as const)
      : ({ height: 200, backgroundColor: "#e5e7eb" } as const);

  const shellStyle = {
    ...(variant === "sidebar" ? ({ flex: 1 } as const) : ({ width: "100%" } as const)),
    backgroundColor: APP_BG,
    borderRadius: 12,
    ...(sticky && Platform.OS === "web" ? ({ position: "sticky", top: 16 } as any) : null),
  };

  const cardStyle = {
    ...(variant === "sidebar" ? ({ flex: 1, minHeight: 0 } as const) : null),
    backgroundColor: CARD_BG,
    borderWidth: 2,
    borderColor: BORDER,
    borderRadius: 12,
    overflow: "hidden",
  };

  return (
    <View style={shellStyle}>
      <Pressable
        accessibilityRole="link"
        accessibilityLabel={`Sponsored: ${activeBanner?.title ?? "Ad"}`}
        onPress={onPress}
        style={({ pressed }) => ({
          ...(variant === "sidebar" ? ({ flex: 1 } as const) : null),
          opacity: pressed ? 0.92 : 1,
          ...(Platform.OS === "web" ? ({ cursor: "pointer" } as any) : null),
        })}
      >
        <View style={cardStyle}>
          {/* Image area */}
          <View style={imageAreaStyle}>
            <Animated.Image
              source={{ uri: activeBanner?.imageUri }}
              resizeMode="cover"
              style={{
                position: "absolute",
                top: 0,
                right: 0,
                bottom: 0,
                left: 0,
                opacity: activeOpacity as any,
              }}
            />
            {len > 1 ? (
              <Animated.Image
                source={{ uri: nextBanner?.imageUri }}
                resizeMode="cover"
                style={{
                  position: "absolute",
                  top: 0,
                  right: 0,
                  bottom: 0,
                  left: 0,
                  opacity: nextOpacity as any,
                }}
              />
            ) : null}

            {/* badge */}
            <View
              style={{
                position: "absolute",
                top: 10,
                left: 10,
                paddingHorizontal: 8,
                paddingVertical: 4,
                borderRadius: 999,
                backgroundColor: "rgba(0,0,0,0.55)",
              }}
            >
              <Text style={{ color: "#ffffff", fontSize: 10, fontWeight: "800", letterSpacing: 0.4 }}>AD</Text>
            </View>
          </View>

          {/* Copy */}
          <View style={{ padding: 12, gap: 6 }}>
            <View style={{ flexDirection: "row", alignItems: "center", justifyContent: "space-between" }}>
              <Text style={{ color: TEXT_DIM, fontSize: 11, fontWeight: "700" }}>{activeBanner?.sponsor ?? "Sponsored"}</Text>
              <Text style={{ color: TEXT_DIM, fontSize: 11, fontWeight: "700" }}>↗</Text>
            </View>

            <Text style={{ color: "#000000", fontSize: 14, fontWeight: "800", lineHeight: 18 }}>
              {activeBanner?.title ?? "Sponsored"}
            </Text>

            <Text style={{ color: TEXT_DIM, fontSize: 12, lineHeight: 16 }}>{activeBanner?.body ?? ""}</Text>

            <View
              style={{
                marginTop: 6,
                alignSelf: "flex-start",
                borderWidth: 2,
                borderColor: BORDER,
                borderRadius: 999,
                paddingHorizontal: 12,
                paddingVertical: 6,
                backgroundColor: "#ffffff",
              }}
            >
              <Text style={{ color: "#000000", fontSize: 12, fontWeight: "800" }}>{activeBanner?.cta ?? "Open"}</Text>
            </View>

            {/* Dots */}
            {len > 1 ? (
              <View style={{ flexDirection: "row", justifyContent: "center", gap: 6, marginTop: 8 }}>
                {banners.map((b, i) => (
                  <View
                    key={b.id}
                    style={{
                      width: 6,
                      height: 6,
                      borderRadius: 999,
                      backgroundColor: i === active ? "#111827" : "#d1d5db",
                    }}
                  />
                ))}
              </View>
            ) : null}

            {activeBanner?.disclaimer ? (
              <Text style={{ color: TEXT_DIM, fontSize: 10, marginTop: 8, lineHeight: 14 }}>
                {activeBanner.disclaimer}
              </Text>
            ) : null}
          </View>
        </View>
      </Pressable>
    </View>
  );
}

function makeGuideRecruitmentBanner(anchorId: string): GuideBanner {
  return {
    id: `guide-ad-${anchorId}`,
    kind: "ad",
    title: "広告掲載を募集中 / Ad space available",
    summary:
      "期間限定で無料掲載中。Google Form からお申し込みください。\nFree for a limited time. Apply via Google Form.",
    url: ADS_FORM_URL,
    imageUri: "https://picsum.photos/seed/goodday-guide-ad/900/650",
    meta: "期間限定無料 • Limited-time free",
    badgeLabel: "AD",
    headerLabel: "ADS / 広告募集",
    ctaLabel: "無料で掲載する / Apply free",
    disclaimer: "Google Form が開きます。 / Opens Google Form.",
  };
}

function buildGuideBanners(guides: GuideItem[], assetBase: string): GuideBanner[] {
  const guideCards = guides.map((guide) => {
    const hero = guide.hero_image
      ? resolveUrl(normalizeWebAssetPath(guide.hero_image), assetBase)
      : "";

    const dateLabel =
      typeof guide.date === "string" && guide.date.trim()
        ? guide.date.trim()
        : typeof guide.published_at === "string" && guide.published_at.trim()
          ? guide.published_at.trim().slice(0, 10)
          : "";

    const meta = [guide.place, dateLabel].filter(Boolean).join(" • ");

    return {
      id: guide.id,
      kind: "guide" as const,
      title: guide.title,
      summary: guide.description || guide.summary || "Open the guide.",
      url: guide.permalink || "./articles/index.html",
      imageUri: hero,
      meta: meta || undefined,
      badgeLabel: "GUIDE",
      headerLabel: "GUIDES",
      ctaLabel: "Open guide",
    };
  });

  if (!guideCards.length || !ADS_FORM_URL) return guideCards;

  const out: GuideBanner[] = [];

  for (let i = 0; i < guideCards.length; i += 1) {
    const guideCard = guideCards[i];
    out.push(guideCard);

    if ((i + 1) % GUIDE_RECRUIT_EVERY_N === 0) {
      out.push(makeGuideRecruitmentBanner(guideCard.id || String(i)));
    }
  }

  return out;
}

function pickInlineGuideBanner(banners: GuideBanner[], anchorId: string): GuideBanner | null {
  if (!banners.length) return null;
  const idx = hashString(anchorId) % banners.length;
  return banners[idx] ?? banners[0] ?? null;
}

function InlineGuideCard({
  banner,
  onPress,
}: {
  banner: GuideBanner;
  onPress: () => void;
}) {
  return (
    <View style={{ width: "100%", backgroundColor: APP_BG, borderRadius: 12 }}>
      <Pressable
        accessibilityRole="link"
        accessibilityLabel={`${banner?.kind === "ad" ? "Ad" : "Guide"}: ${banner?.title ?? "Guide"}`}
        onPress={onPress}
        style={({ pressed }) => ({
          opacity: pressed ? 0.92 : 1,
          ...(Platform.OS === "web" ? ({ cursor: "pointer" } as any) : null),
        })}
      >
        <View
          style={{
            backgroundColor: CARD_BG,
            borderWidth: 2,
            borderColor: BORDER,
            borderRadius: 12,
            overflow: "hidden",
          }}
        >
          <View style={{ height: 200, backgroundColor: "#e5e7eb" }}>
            {banner?.imageUri ? (
              <Image
                source={{ uri: banner.imageUri }}
                resizeMode="cover"
                style={{ position: "absolute", top: 0, right: 0, bottom: 0, left: 0 }}
              />
            ) : null}

            <View
              style={{
                position: "absolute",
                top: 10,
                left: 10,
                paddingHorizontal: 8,
                paddingVertical: 4,
                borderRadius: 999,
                backgroundColor: "rgba(0,0,0,0.55)",
              }}
            >
              <Text style={{ color: "#ffffff", fontSize: 10, fontWeight: "800", letterSpacing: 0.4 }}>
                {banner?.badgeLabel ?? "GUIDE"}
              </Text>
            </View>
          </View>

          <View style={{ padding: 12, gap: 6 }}>
            <View style={{ flexDirection: "row", alignItems: "center", justifyContent: "space-between" }}>
              <Text style={{ color: TEXT_DIM, fontSize: 11, fontWeight: "700" }}>
                {banner?.headerLabel ?? "GUIDES"}
              </Text>
              <Text style={{ color: TEXT_DIM, fontSize: 11, fontWeight: "700" }}>↗</Text>
            </View>

            <Text style={{ color: "#000000", fontSize: 16, fontWeight: "800", lineHeight: 22 }} numberOfLines={3}>
              {banner?.title ?? "Guide"}
            </Text>

            {banner?.meta ? (
              <Text style={{ color: TEXT_DIM, fontSize: 11, fontWeight: "700", lineHeight: 16 }} numberOfLines={1}>
                {banner.meta}
              </Text>
            ) : null}

            <Text style={{ color: TEXT_DIM, fontSize: 12, lineHeight: 18 }} numberOfLines={4}>
              {banner?.summary ?? ""}
            </Text>

            <View
              style={{
                marginTop: 6,
                alignSelf: "flex-start",
                borderWidth: 2,
                borderColor: BORDER,
                borderRadius: 999,
                paddingHorizontal: 12,
                paddingVertical: 6,
                backgroundColor: "#ffffff",
              }}
            >
              <Text style={{ color: "#000000", fontSize: 12, fontWeight: "800" }}>
                {banner?.ctaLabel ?? "Open guide"}
              </Text>
            </View>

            {banner?.disclaimer ? (
              <Text style={{ color: TEXT_DIM, fontSize: 10, marginTop: 8, lineHeight: 14 }}>
                {banner.disclaimer}
              </Text>
            ) : null}
          </View>
        </View>
      </Pressable>
    </View>
  );
}

function GuideSidebar({
  banners,
  active,
  next,
  progress,
}: {
  banners: GuideBanner[];
  active: number;
  next: number;
  progress: Animated.Value;
}) {
  const len = banners.length;

  if (!banners.length) {
    return (
      <View
        style={{
          minHeight: 0,
          height: "100%",
          borderWidth: 2,
          borderColor: BORDER,
          borderRadius: 20,
          backgroundColor: "#ffffff",
          padding: 14,
        }}
      >
        <Text style={{ fontSize: 20, fontWeight: "800", color: "#000", marginBottom: 12 }}>
          Guide
        </Text>
        <Text style={{ color: TEXT_DIM, fontSize: 13, lineHeight: 18 }}>
          No guides yet.
        </Text>
      </View>
    );
  }

  const safeActive = Math.max(0, Math.min(active, len - 1));
  const safeNext = Math.max(0, Math.min(next, len - 1));

  const activeOpacity =
    len <= 1
      ? 1
      : progress.interpolate({
          inputRange: [0, 1],
          outputRange: [1, 0],
        });

  const nextOpacity = len <= 1 ? 0 : progress;

  const activeBanner = banners[safeActive] ?? banners[0];
  const nextBanner = banners[safeNext] ?? banners[0];

  return (
    <View
      style={{
        flex: 1,
        backgroundColor: APP_BG,
        borderRadius: 12,
        ...(Platform.OS === "web" ? ({ position: "sticky", top: 16 } as any) : null),
      }}
    >
      <Pressable
        accessibilityRole="link"
        accessibilityLabel={`${activeBanner?.kind === "ad" ? "Ad" : "Guide"}: ${activeBanner?.title ?? "Guide"}`}
        onPress={() => openResolvedUrl(activeBanner?.url || "./articles/index.html")}
        style={({ pressed }) => ({
          flex: 1,
          opacity: pressed ? 0.92 : 1,
          ...(Platform.OS === "web" ? ({ cursor: "pointer" } as any) : null),
        })}
      >
        <View
          style={{
            flex: 1,
            minHeight: 0,
            backgroundColor: CARD_BG,
            borderWidth: 2,
            borderColor: BORDER,
            borderRadius: 12,
            overflow: "hidden",
          }}
        >
          <View style={{ flex: 1, minHeight: 0, backgroundColor: "#e5e7eb" }}>
            {activeBanner?.imageUri ? (
              <Animated.Image
                source={{ uri: activeBanner.imageUri }}
                resizeMode="cover"
                style={{
                  position: "absolute",
                  top: 0,
                  right: 0,
                  bottom: 0,
                  left: 0,
                  opacity: activeOpacity as any,
                }}
              />
            ) : null}
            {len > 1 && nextBanner?.imageUri ? (
              <Animated.Image
                source={{ uri: nextBanner.imageUri }}
                resizeMode="cover"
                style={{
                  position: "absolute",
                  top: 0,
                  right: 0,
                  bottom: 0,
                  left: 0,
                  opacity: nextOpacity as any,
                }}
              />
            ) : null}

            <View
              style={{
                position: "absolute",
                top: 10,
                left: 10,
                paddingHorizontal: 8,
                paddingVertical: 4,
                borderRadius: 999,
                backgroundColor: "rgba(0,0,0,0.55)",
              }}
            >
              <Text style={{ color: "#ffffff", fontSize: 10, fontWeight: "800", letterSpacing: 0.4 }}>
                {activeBanner?.badgeLabel ?? "GUIDE"}
              </Text>
            </View>
          </View>

          <View style={{ padding: 12, gap: 6 }}>
            <View style={{ flexDirection: "row", alignItems: "center", justifyContent: "space-between" }}>
              <Text style={{ color: TEXT_DIM, fontSize: 11, fontWeight: "700" }}>
                {activeBanner?.headerLabel ?? "GUIDES"}
              </Text>
              <Text style={{ color: TEXT_DIM, fontSize: 11, fontWeight: "700" }}>↗</Text>
            </View>

            <Text style={{ color: "#000000", fontSize: 18, fontWeight: "800", lineHeight: 24 }} numberOfLines={3}>
              {activeBanner?.title ?? "Guide"}
            </Text>

            {activeBanner?.meta ? (
              <Text style={{ color: TEXT_DIM, fontSize: 11, fontWeight: "700", lineHeight: 16 }}>
                {activeBanner.meta}
              </Text>
            ) : null}

            <Text style={{ color: TEXT_DIM, fontSize: 12, lineHeight: 18 }} numberOfLines={5}>
              {activeBanner?.summary ?? ""}
            </Text>

            <View
              style={{
                marginTop: 6,
                alignSelf: "flex-start",
                borderWidth: 2,
                borderColor: BORDER,
                borderRadius: 999,
                paddingHorizontal: 12,
                paddingVertical: 6,
                backgroundColor: "#ffffff",
              }}
            >
              <Text style={{ color: "#000000", fontSize: 12, fontWeight: "800" }}>
                {activeBanner?.ctaLabel ?? "Open guide"}
              </Text>
            </View>

            {activeBanner?.disclaimer ? (
              <Text style={{ color: TEXT_DIM, fontSize: 10, lineHeight: 14 }}>
                {activeBanner.disclaimer}
              </Text>
            ) : null}

            {len > 1 ? (
              <View style={{ flexDirection: "row", justifyContent: "center", gap: 6, marginTop: 8 }}>
                {banners.map((banner, i) => (
                  <View
                    key={banner.id}
                    style={{
                      width: 6,
                      height: 6,
                      borderRadius: 999,
                      backgroundColor: i === safeActive ? "#111827" : "#d1d5db",
                    }}
                  />
                ))}
              </View>
            ) : null}
          </View>
        </View>
      </Pressable>
    </View>
  );
}

function Slot({ side }: { side: "left" | "right" }) {
  const enabled = process.env.EXPO_PUBLIC_USE_SLOT === "1";
  if (!enabled) return null;

  const banners = SLOT_BANNERS;
  if (!banners.length) return null;

  // Offset the starting banner so L/R columns don't look identical.
  const startIndex = useMemo(() => {
    if (banners.length <= 1) return 0;
    const base = side === "right" ? 2 : 0;
    return base % banners.length;
  }, [banners.length, side]);

  return <SlotCard 
    banners={banners} 
    startIndex={startIndex} 
    sticky variant="sidebar" />;
}
export default function HomeScreen() {
  const listRef = useRef<FlatList<TimelineItem>>(null);
  const deepLinkAttemptsRef = useRef<number>(0);
  const [deepLinkPostId, setDeepLinkPostId] = useState<string | null>(null);
  const FEED_URL = (process.env.EXPO_PUBLIC_FEED_URL || process.env.EXPO_PUBLIC_FEED_JSON_URL || "./latest.json").trim();
  const ASSET_BASE_URL = (process.env.EXPO_PUBLIC_ASSET_BASE_URL || "").trim();
  const FEED_BASE_URL = (process.env.EXPO_PUBLIC_FEED_BASE_URL || ASSET_BASE_URL || "").trim();
  const IMAGE_BASE_URL = (process.env.EXPO_PUBLIC_IMAGE_BASE_URL || ASSET_BASE_URL || "").trim();
  const SHARE_SD_INDEX_URL = (process.env.EXPO_PUBLIC_SHARE_SD_INDEX_URL || "").trim();
  const GUIDES_INDEX_URL = (process.env.EXPO_PUBLIC_LONGFORM_GUIDES_INDEX_URL || "./articles/index.json").trim();
  const { width } = useWindowDimensions();
  const showSidebars = width >= 980;

  const RESOLVED_FEED_URL = useMemo(() => {
    const normalized = normalizeWebAssetPath(FEED_URL);
    const base = FEED_BASE_URL || (typeof window !== "undefined" ? window.location.href : normalized);
    try {
      if (normalized.startsWith("http://") || normalized.startsWith("https://")) return normalized;
      if (base) return new URL(normalized, base).toString();
    } catch {
      // ignore
    }
    return normalized;
  }, [FEED_URL, FEED_BASE_URL]);

  const [feed, setFeed] = useState<Feed | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [refreshing, setRefreshing] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const [nextUrl, setNextUrl] = useState<string | null>(null);
  const [loadingMore, setLoadingMore] = useState<boolean>(false);

  const [shareSdIndex, setShareSdIndex] = useState<ShareSdIndex | null>(null);
  const [guidesIndex, setGuidesIndex] = useState<GuidesIndex | null>(null);

  const fetchJson = useCallback(async (url: string): Promise<{ raw: string; parsed: unknown }> => {
    const finalUrl = addCacheBuster(url);
    const res = await fetch(finalUrl, { headers: { "Cache-Control": "no-cache" } });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const raw = await res.text();
    const parsed = safeJsonParse(raw);
    return { raw, parsed };
  }, []);
  
  const sortedItems = useMemo(() => {
    const items = feed?.items ?? [];
    return [...items].sort((a, b) => {
      const ta = (a.generated_at || a.date || "").toString();
      const tb = (b.generated_at || b.date || "").toString();
      return ta < tb ? 1 : ta > tb ? -1 : 0;
    });
  }, [feed]);

  const timelineItems = useMemo(() => interleaveAds(sortedItems), [sortedItems]);

  const [effectiveUrl, setEffectiveUrl] = useState<string>(RESOLVED_FEED_URL);

  // keep base url in sync when FEED_URL changes
  useEffect(() => {
    setEffectiveUrl(RESOLVED_FEED_URL);
  }, [RESOLVED_FEED_URL]);

  // Read ?post=<id> on web
  useEffect(() => {
    if (Platform.OS !== "web" || typeof window === "undefined") return;
    try {
      const sp = new URLSearchParams(window.location.search);
      const pid = sp.get("post");
      if (pid) setDeepLinkPostId(pid);
    } catch {}
  }, []);

useEffect(() => {
  if (!SHARE_SD_INDEX_URL) return;

  let cancelled = false;

  (async () => {
    try {
      const base =
        FEED_BASE_URL || (Platform.OS === "web" && typeof window !== "undefined" ? window.location.href : RESOLVED_FEED_URL);

      const resolved = resolveUrl(normalizeWebAssetPath(SHARE_SD_INDEX_URL), base);
      const target = await fetchJson(resolved);
      const normalized = normalizeShareSdIndex(target.parsed);

      if (!cancelled) {
        setShareSdIndex(normalized);
      }
    } catch {
      // ignore (images are optional)
    }
  })();

  return () => {
    cancelled = true;
  };
}, [SHARE_SD_INDEX_URL, FEED_BASE_URL, RESOLVED_FEED_URL, fetchJson]);

useEffect(() => {
  if (!GUIDES_INDEX_URL) return;

  let cancelled = false;

  (async () => {
    try {
      const base =
        ASSET_BASE_URL ||
        (Platform.OS === "web" && typeof window !== "undefined" ? window.location.href : RESOLVED_FEED_URL);

      const resolved = resolveUrl(normalizeWebAssetPath(GUIDES_INDEX_URL), base);
      const target = await fetchJson(resolved);
      const normalized = normalizeGuidesIndex(target.parsed);

      if (!cancelled) {
        setGuidesIndex(normalized);
      }
    } catch {
      if (!cancelled) {
        setGuidesIndex(null);
      }
    }
  })();

  return () => {
    cancelled = true;
  };
}, [GUIDES_INDEX_URL, ASSET_BASE_URL, RESOLVED_FEED_URL, fetchJson]);

const sharePromptToImage = useMemo(() => {
  const m = new Map<string, string>();
  for (const it of shareSdIndex?.items ?? []) {
    if (it.prompt && it.image) {
      m.set(it.prompt, it.image);
      continue;
    }
    if (it.date && it.place && it.image) {
      m.set(`${it.date}|${it.place}`, it.image);
    }
  }
  return m;
}, [shareSdIndex]);

const assetBase = useMemo(() => {
  if (IMAGE_BASE_URL) return IMAGE_BASE_URL;
  if (ASSET_BASE_URL) return ASSET_BASE_URL;
  if (Platform.OS === "web" && typeof window !== "undefined") return window.location.href;
  return effectiveUrl || RESOLVED_FEED_URL;
}, [IMAGE_BASE_URL, ASSET_BASE_URL, effectiveUrl, RESOLVED_FEED_URL]);

const visibleGuides = useMemo(() => guidesIndex?.items ?? [], [guidesIndex]);
const guideBanners = useMemo(() => buildGuideBanners(visibleGuides, assetBase), [visibleGuides, assetBase]);
const guideLen = guideBanners.length;
const [activeGuide, setActiveGuide] = useState(0);
const guideProgress = useRef(new Animated.Value(0)).current;

useEffect(() => {
  if (activeGuide >= guideLen) setActiveGuide(0);
}, [activeGuide, guideLen]);

useEffect(() => {
  if (guideLen <= 1) return;

  let cancelled = false;

  const interval = setInterval(() => {
    const nextGuide = (activeGuide + 1) % guideLen;

    guideProgress.stopAnimation();
    guideProgress.setValue(0);

    Animated.timing(guideProgress, {
      toValue: 1,
      duration: SLOT_FADE_MS,
      useNativeDriver: Platform.OS !== "web",
    }).start(({ finished }) => {
      if (!finished || cancelled) return;
      setActiveGuide(nextGuide);
      guideProgress.setValue(0);
    });
  }, SLOT_ROTATE_MS);

  return () => {
    cancelled = true;
    clearInterval(interval);
    guideProgress.stopAnimation();
  };
}, [activeGuide, guideLen, guideProgress]);

const leftGuideActiveIndex  = guideLen <= 1 ? 0 : activeGuide % guideLen;
const leftGuideNextIndex    = guideLen <= 1 ? 0 : (leftGuideActiveIndex + 1) % guideLen;
const rightGuideActiveIndex = guideLen <= 1 ? 0 : (leftGuideActiveIndex + Math.floor(guideLen / 2)) % guideLen;
const rightGuideNextIndex   = guideLen <= 1 ? 0 : guideLen === 2 ? leftGuideActiveIndex : (rightGuideActiveIndex + 1) % guideLen;

const getImageUrisForItem = useCallback(
  (item: FeedItem): string[] => {
    const uris: string[] = [];

    const push = (p?: string) => {
      const s = String(p ?? "").trim();
      if (!s) return;
      const resolved = resolveUrl(normalizeWebAssetPath(s), assetBase);
      if (!uris.includes(resolved)) uris.push(resolved);
    };

    // 1) Direct field (best)
    if (item.image) push(item.image);

    // 2) Stem-match rule: if id looks like feed stem, try /image/<id>.png
    const id = String(item.id ?? "").trim();
    if (id && id.startsWith("feed_")) {
      push(`/image/${encodeURIComponent(id)}.png`);
    }

    // 3) Optional: share_sd index match (if configured)
    const place = item.place || feed?.place;
    const prompt = item.image_prompt || buildSharePrompt(item.text, place);

    const fromPrompt = sharePromptToImage.get(prompt);
    if (fromPrompt) push(fromPrompt);

    if (item.date && place) {
      const byKey = sharePromptToImage.get(`${item.date}|${place}`);
      if (byKey) push(byKey);
    }

    return uris;
  },
  [assetBase, feed?.place, sharePromptToImage],
);

  useEffect(() => {
    ensureWebScrollbarStyle();
  }, []);

  const load = useCallback(
  async () => {
    let currentEffectiveUrl = RESOLVED_FEED_URL;

    try {
      setError(null);
      setNextUrl(null);

      const base =
        FEED_BASE_URL || (Platform.OS === "web" && typeof window !== "undefined" ? window.location.href : RESOLVED_FEED_URL);

      // Try the configured URL first, then common fallbacks (root and /feed/).
      const candidates = Array.from(
        new Set([
          RESOLVED_FEED_URL,
          resolveUrl("./latest.json", base),
          resolveUrl("./feed/latest.json", base),
          resolveUrl("./feed/latest.json", base),
          resolveUrl("./feed/feed.json", base),
          resolveUrl("./feed.json", base),
          resolveUrl("./output.json", base),
          resolveUrl("./feed/output.json", base),
        ]),
      );

      let firstUrl = "";
      let first: { raw: string; parsed: unknown } | null = null;
      let lastErr: any = null;

      for (const u of candidates) {
        try {
          first = await fetchJson(u);
          firstUrl = u;
          break;
        } catch (e: any) {
          lastErr = e;
        }
      }

      if (!first) throw lastErr ?? new Error("Failed to load feed");

      setEffectiveUrl(firstUrl);

      const pointer = getFeedPointer(first.parsed);
      let target = first;
      let baseForPointers = firstUrl;

      if (pointer) {
        currentEffectiveUrl = resolveUrl(pointer, baseForPointers);
        setEffectiveUrl(currentEffectiveUrl);
        target = await fetchJson(currentEffectiveUrl);
        baseForPointers = currentEffectiveUrl;
      }

      const normalized = normalizeFeed(target.parsed);
      if (!normalized) {
        const preview = first.raw.slice(0, 180).replace(/\s+/g, " ").trim();
        throw new Error(`Invalid feed JSON shape\nURL: ${currentEffectiveUrl}\nRAW: ${preview}`);
      }

      const nextPointer = getNextPointer(target.parsed);
      setNextUrl(nextPointer ? resolveUrl(nextPointer, baseForPointers) : null);

      setFeed(normalized);
    } catch (e: any) {
      setError(e?.message ?? "Failed to load feed");
      setFeed(null);
    } finally {
      setLoading(false);
    }
  },
  [FEED_BASE_URL, RESOLVED_FEED_URL, fetchJson],
);

  useEffect(() => {
    void load();
  }, [load]);

  const onRefresh = useCallback(async () => {
    setRefreshing(true);
    await load();
    setRefreshing(false);
  }, [load]);

    const loadMore = useCallback(async () => {
      if (!nextUrl || loadingMore) return;
  
      const pageUrl = nextUrl;
      setLoadingMore(true);
  
      try {
        const target = await fetchJson(pageUrl);
        const normalized = normalizeFeed(target.parsed);
        if (!normalized) {
          const preview = target.raw.slice(0, 180).replace(/\s+/g, " ").trim();
          throw new Error(`Invalid feed JSON shape\nURL: ${pageUrl}\nRAW: ${preview}`);
        }
  
        const nextPointer = getNextPointer(target.parsed);
        setNextUrl(nextPointer ? resolveUrl(nextPointer, pageUrl) : null);
  
        setFeed((prev) => {
          const prevItems = prev?.items ?? [];
          const merged: FeedItem[] = [...prevItems];
          const seen = new Set(prevItems.map((it) => it.id));
  
          for (const it of normalized.items) {
            if (!seen.has(it.id)) {
              merged.push(it);
              seen.add(it.id);
            }
          }
  
          return {
            updated_at: prev?.updated_at ?? normalized.updated_at,
            place: prev?.place ?? normalized.place,
            items: merged,
          };
        });
      } catch (e: any) {
        setError(e?.message ?? "Failed to load more");
      } finally {
        setLoadingMore(false);
      }
    }, [fetchJson, loadingMore, nextUrl]);


  const openFeed = useCallback(() => {
    if (!effectiveUrl) return;
    if (Platform.OS !== "web") return;
    void Linking.openURL(effectiveUrl);
  }, [effectiveUrl]);

  const Header = (
    <View style={{ padding: 0, gap: 0 }}>
      {error ? (
        <View
          style={{
            backgroundColor: "#7f1d1d",
            borderRadius: 14,
            padding: 12,
          }}
        >
          <Text style={{ color: "#000000", fontWeight: "800" }}>Error</Text>
          <Text style={{ color: "#000000", marginTop: 6 }}>{error}</Text>
        </View>
      ) : null}
    </View>
  );

  if (loading) {
    return (
      <View style={{ flex: 1, backgroundColor: APP_BG, alignItems: "center", justifyContent: "center", padding: 16 }}>
        <ActivityIndicator />
        <Text style={{ marginTop: 10, color: TEXT_DIM }}>Loading…</Text>
      </View>
    );
  }

  const list = (
    <FlatList
      ref={listRef}
      nativeID={FEED_SCROLL_ID}
      showsVerticalScrollIndicator={false}
      style={{ flex: 1, backgroundColor: APP_BG }}
      contentContainerStyle={{ paddingBottom: 18 }}
      data={timelineItems}
      onScrollToIndexFailed={(info) => {
        listRef.current?.scrollToOffset({
          offset: info.averageItemLength * info.index,
          animated: false,
        });
        setTimeout(() => {
          listRef.current?.scrollToIndex({ index: info.index, animated: false });
        }, 50);
      }}
      keyExtractor={(it) => it.id}
      ListHeaderComponent={Header}
      refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} />}
      onEndReached={loadMore}
      onEndReachedThreshold={0.5}
            ListFooterComponent={
              loadingMore ? (
                <View style={{ padding: 16, alignItems: "center" }}>
                  <ActivityIndicator />
                  <Text style={{ marginTop: 8, color: TEXT_DIM }}>Loading older posts…</Text>
                </View>
              ) : nextUrl ? (
                <View style={{ padding: 16, alignItems: "center" }}>
                  <Text style={{ color: TEXT_DIM }}>Scroll to load older posts…</Text>
                </View>
              ) : (feed?.items?.length ?? 0) > 0 ? (
                <View style={{ padding: 16, alignItems: "center" }}>
                  <Text style={{ color: TEXT_DIM }}>No more posts.</Text>
                </View>
              ) : null
            }
      renderItem={({ item }) => {
        if (isSlotItem(item)) {
          const banner = pickInlineGuideBanner(guideBanners, item.id);

          if (banner) {
            return (
              <View style={{ paddingHorizontal: 16, paddingBottom: 12 }}>
                <View style={{ flexDirection: "row", alignItems: "flex-start" }}>
                  {/* keep alignment with the mascot column */}
                  <View style={{ flex: 1 }}>
                    <InlineGuideCard banner={banner} onPress={() => openResolvedUrl(banner.url)} />
                  </View>
                </View>
              </View>
            );
          }

          const open = () => {
            if (!item.url) return;
            openResolvedUrl(item.url);
          };

          return (
            <Pressable onPress={open}>
              <View 
                style={{ 
                  paddingHorizontal: 16, 
                  paddingBottom: 12 
                  }}>
                <View style={{ flexDirection: "row", alignItems: "flex-start" }}>
                  <View style={{ flex: 1 }}>
                    {/* Speech-bubble wrapper */}
                    <View style={{ position: "relative", marginTop: 2 }}>
                      {/* ✅ 1) Bubble body FIRST */}
                      <View
                        style={{
                          backgroundColor: ITEM_BG,
                          padding: 12,
                          borderRadius: BUBBLE_RADIUS,
                          borderWidth: BUBBLE_BORDER_W,
                          borderColor: BORDER,
                          minHeight: MASCOT_SIZE,
                          shadowColor: "#000000",
                          shadowOffset: { width: 0, height: 2 },
                          shadowOpacity: 0.12,
                          shadowRadius: 6,
                          elevation: 2,
                          zIndex: 1,
                        }}
                      >
                        <View style={{ flexDirection: "row", flexWrap: "wrap", alignItems: "center", gap: 8 }}>
                          <Text style={{ color: "#000000", fontWeight: "800" }}>{item.title}</Text>
                          {item.sponsor ? <Text style={{ color: TEXT_DIM }}>• {item.sponsor}</Text> : null}
                        </View>
                        <Text style={{ color: "#000000", marginTop: 8, fontSize: 16, lineHeight: 22 }}>{item.body}</Text>
                      </View>
                    </View>
                  </View>
                </View>
              </View>
            </Pressable>
          );
        }

        const imageUris = getImageUrisForItem(item);
        return (
          <View style={{ paddingHorizontal: 16, paddingBottom: 12 }}>
            <View style={{ flexDirection: "row", alignItems: "flex-start" }}>
              <View style={{ width: MASCOT_COL_W, alignItems: "center" }}>
                <View style={{ marginTop: 2 }}>
                  <Mascot uri={getMascotUriForItem(item, assetBase)} assetBase={assetBase} />
                </View>
              </View>

              <View style={{ flex: 1 }}>
                {/* Speech-bubble wrapper */}
                <View style={{ position: "relative", marginTop: 2 }}>
                  {/* ✅ 1) Bubble body FIRST */}
                  <View
                    style={{
                      backgroundColor: CARD_BG,
                      padding: 12,
                      borderRadius: BUBBLE_RADIUS,
                      borderWidth: BUBBLE_BORDER_W,
                      borderColor: BORDER,
                      minHeight: MASCOT_SIZE,
                      shadowColor: "#000000",
                      shadowOffset: { width: 0, height: 2 },
                      shadowOpacity: 0.12,
                      shadowRadius: 6,
                      elevation: 2,
                      zIndex: 1,
                    }}
                  >
                    <View style={{ flexDirection: "row", flexWrap: "wrap", alignItems: "center", gap: 8 }}>
                      {item.generated_at ? <Text style={{ color: TEXT_DIM }}>{formatJst(item.generated_at, true)}</Text> : null}
                    </View>
                    
                    <FeedBubbleImage uris={imageUris} />
                    <Text style={{ color: "#000000", marginTop: 8, fontSize: 16, lineHeight: 22 }}>{item.text}</Text>

                    {Array.isArray(item.links) && item.links.length > 0 ? (
                      <View style={{ flexDirection: "row", flexWrap: "wrap", gap: 6, marginTop: 8 }}>
                        {item.links.slice(0, 2).map((link, idx) => (
                          <Pressable
                            key={`${link.url}-${idx}`}
                            onPress={() => openResolvedUrl(link.url)}
                          >
                            <Text style={{ color: "#0B57D0", textDecorationLine: "underline", fontSize: 12 }}>
                              🔗 {link.title || link.url}
                            </Text>
                          </Pressable>
                        ))}
                      </View>
                    ) : null}
                  </View>

                  {/* ✅ 2) Tail AFTER (on top) to cover the bubble border line */}
                  <View
                    pointerEvents="none"
                    style={{
                      position: "absolute",
                      left: -7,
                      top: 22,
                      width: 14,
                      height: 14,
                      backgroundColor: CARD_BG,
                      transform: [{ rotate: "45deg" }],
                      borderLeftWidth: BUBBLE_BORDER_W,
                      borderBottomWidth: BUBBLE_BORDER_W,
                      borderColor: BORDER,
                      zIndex: 10,
                      elevation: 3,
                    }}
                  />
                </View>
              </View>
            </View>
          </View>
        );
      }}
      ListEmptyComponent={
        <View style={{ padding: 16 }}>
          <Text style={{ color: TEXT_DIM }}>No posts yet.</Text>
        </View>
      }
    />
  );

  if (!showSidebars) {
    return list;
  }

  return (
    <View
      style={{
        flex: 1,
        padding: 6,
        flexDirection: "row",
        backgroundColor: APP_BG,
        gap: 12,
        alignItems: "stretch",
      }}
    >
      {/* Left sidebar: rotating guide card */}
      <View style={{ flex: 1, minWidth: SIDEBAR_W, minHeight: 0 }}>
        <GuideSidebar
          banners={guideBanners}
          active={leftGuideActiveIndex}
          next={leftGuideNextIndex}
          progress={guideProgress}
        />
      </View>

      {/* Center: keep 760 as the “target” width, but allow shrinking */}
      <View style={{ width: CONTENT_MAX_W, minWidth: 0, flexShrink: 1 }}>
        {list}
      </View>

      {/* Right sidebar: rotating guide card, always offset from left */}
      <View style={{ flex: 1, minWidth: SIDEBAR_W, minHeight: 0 }}>
        <GuideSidebar
          banners={guideBanners}
          active={rightGuideActiveIndex}
          next={rightGuideNextIndex}
          progress={guideProgress}
        />
      </View>
    </View>
  );
}