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

type FeedItem = {
  id: string;
  date: string; // YYYY-MM-DD
  text: string;
  place?: string;
  generated_at?: string; // ISO string (often Z)
  image?: string; // local path or absolute URL
  image_prompt?: string; // optional (for matching)
  permalink?: string;
  links?: string[];
};

type Feed = {
  updated_at?: string;
  place?: string;
  items: FeedItem[];
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
const MASCOT_RADIUS = 12;
const MASCOT_BORDER_W = 2;
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
    url: RAW_CONTACT_URL,
    sponsor: "demo1",
    disclaimer: "demo1",
    emoji: "🧜‍♀️",
  },
  {
    title: "demo1",
    body: "demo1",
    cta: "check",
    url: RAW_CONTACT_URL,
    sponsor: "demo1",
    disclaimer: "demo1",
    emoji: "🧜‍♀️",
  },
  {
    title: "demo1",
    body: "demo1",
    cta: "check",
    url: RAW_CONTACT_URL,
    sponsor: "demo1",
    disclaimer: "demo1",
    emoji: "🧜‍♀️",
  },
  {
    title: "demo1",
    body: "demo1",
    cta: "check",
    url: RAW_CONTACT_URL,
    sponsor: "demo1",
    disclaimer: "demo1",
    emoji: "🧜‍♀️",
  },
  {
    title: "demo1",
    body: "demo1",
    cta: "check",
    url: RAW_CONTACT_URL,
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

const SLOT_ROTATE_MS = Math.max(2500, Number((process.env.EXPO_PUBLIC_SLOT_ROTATE_MS || "6500").trim()) || 6500);
const SLOT_FADE_MS = Math.max(200, Number((process.env.EXPO_PUBLIC_SLOT_FADE_MS || "800").trim()) || 800);

const SLOT_BANNERS: SlotBanner[] = [
  {
    id: "slot-0",
    title: "Ocean view, zero effort",
    body: "",
    cta: "Open demo",
    url: RAW_CONTACT_URL,
    imageUri: "https://picsum.photos/seed/goodday_ocean/900/650",
    sponsor: "GOODDAY",
    disclaimer: "Demo ad slot — not a real promotion.",
  },
  {
    id: "slot-1",
    title: "Coffee & quiet time",
    body: "",
    cta: "See more",
    url: RAW_CONTACT_URL,
    imageUri: "https://picsum.photos/seed/goodday_coffee/900/650",
    sponsor: "GOODDAY",
    disclaimer: "Demo ad slot — not a real promotion.",
  },
  {
    id: "slot-2",
    title: "Weekend micro trip",
    body: "",
    cta: "View route",
    url: RAW_CONTACT_URL,
    imageUri: "https://picsum.photos/seed/goodday_trip/900/650",
    sponsor: "GOODDAY",
    disclaimer: "Demo ad slot — not a real promotion.",
  },
  {
    id: "slot-3",
    title: "Sunset soundtrack",
    body: "",
    cta: "Play",
    url: RAW_CONTACT_URL,
    imageUri: "https://picsum.photos/seed/goodday_sunset/900/650",
    sponsor: "GOODDAY",
    disclaimer: "Demo ad slot — not a real promotion.",
  },
  {
    id: "slot-4",
    title: "Mountain air",
    body: "",
    cta: "Learn more",
    url: RAW_CONTACT_URL,
    imageUri: "https://picsum.photos/seed/goodday_mountain/900/650",
    sponsor: "GOODDAY",
    disclaimer: "Demo ad slot — not a real promotion.",
  },
  {
    id: "slot-5",
    title: "City lights",
    body: "",
    cta: "Open",
    url: RAW_CONTACT_URL,
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
  // "5件に1件" = every Nth item is an ad (i.e. after N-1 posts)
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
          const permalink = typeof it?.permalink === "string" ? it.permalink : undefined;
          return { id, date, text, place, generated_at, image, image_prompt, permalink };
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
      const updated_at = generated_at;
      const permalink = typeof obj?.permalink === "string" ? obj.permalink : undefined;
      return { updated_at, place, items: [{ id, date, text, place, generated_at, image, image_prompt, permalink }] };
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
          const permalink = typeof it?.permalink === "string" ? it.permalink : undefined;
          return { id, date, text, place, generated_at, image, image_prompt, permalink };
        })
      .filter(Boolean) as FeedItem[];

    const last = parsed.length > 0 ? (parsed[parsed.length - 1] as any) : null;
    const updated_at = typeof last?.generated_at === "string" ? last.generated_at : undefined;
    const place = typeof last?.place === "string" ? last.place : undefined;

    return { updated_at, place, items };
  }

  return null;
}

function buildPermalink(id: string, permalink?: string): string {
  // If backend provided a relative permalink, make it absolute on web for easy sharing.
  if (Platform.OS === "web" && typeof window !== "undefined") {
    const u = new URL(window.location.href);
    u.searchParams.delete("redirect");
    u.searchParams.set("post", id);
    u.hash = "";
    return u.toString();
  }
  // Fallback (native / unknown): keep relative
  return permalink ?? `./?post=${encodeURIComponent(id)}`;
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

function addCacheBuster(url: string): string {
  const sep = url.includes("?") ? "&" : "?";
  return `${url}${sep}v=${Date.now()}`;
}

function Mascot({ size = MASCOT_SIZE }: { size?: number }) {
  const [failed, setFailed] = useState(false);
  const envUri = (process.env.EXPO_PUBLIC_MASCOT_URI || "").trim();

  const resolvedEnvUri = useMemo(() => {
    if (!envUri) return "";
    // Only use the env URI when it is clearly an absolute URL/data URI.
    // (Relative strings like "avatar.png" often cause 404s on GitHub Pages.)
    if (/^(https?:)?\/\//i.test(envUri) || envUri.startsWith("data:")) return envUri;
    return "";
  }, [envUri]);

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
      }}
      accessibilityLabel="Mascot"
    >
      {children}
    </View>
  );

  if (!failed && resolvedEnvUri) {
    return (
      <Frame>
        <Image
          source={{ uri: resolvedEnvUri }}
          style={{ width: "100%", height: "100%" }}
          accessibilityLabel="Mascot"
          onError={() => setFailed(true)}
        />
        />
      </Frame>
    );
  }

  try {
    const fallback = require("../assets/images/avatar.png");
    return (
      <Frame>
        <Image source={fallback} style={{ width: "100%", height: "100%" }} accessibilityLabel="Mascot" />
      </Frame>
    );
  } catch {
    // ignore
  }

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
  const FEED_URL = (process.env.EXPO_PUBLIC_FEED_URL || "./latest.json").trim();
  const SHARE_SD_INDEX_URL = (process.env.EXPO_PUBLIC_SHARE_SD_INDEX_URL || "").trim();
  const { width } = useWindowDimensions();
  const showSidebars = width >= 980;

  const RESOLVED_FEED_URL = useMemo(() => {

    const normalized = normalizeWebAssetPath(FEED_URL);
     try {
       if (normalized.startsWith("http://") || normalized.startsWith("https://")) return normalized;
       if (typeof window !== "undefined") return new URL(normalized, window.location.href).toString();
     } catch {
       // ignore
     }
     return normalized;
   }, [FEED_URL]);

  const [feed, setFeed] = useState<Feed | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [refreshing, setRefreshing] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const [nextUrl, setNextUrl] = useState<string | null>(null);
  const [loadingMore, setLoadingMore] = useState<boolean>(false);

  const [shareSdIndex, setShareSdIndex] = useState<ShareSdIndex | null>(null);

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
        Platform.OS === "web" && typeof window !== "undefined" ? window.location.href : RESOLVED_FEED_URL;

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
}, [SHARE_SD_INDEX_URL, RESOLVED_FEED_URL, fetchJson]);

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
  if (Platform.OS === "web" && typeof window !== "undefined") return window.location.href;
  return effectiveUrl || RESOLVED_FEED_URL;
}, [effectiveUrl, RESOLVED_FEED_URL]);

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
        Platform.OS === "web" && typeof window !== "undefined" ? window.location.href : RESOLVED_FEED_URL;

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
  [RESOLVED_FEED_URL, fetchJson],
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


  // Jump to permalink target (web only). Auto-page older items until found (bounded).
  useEffect(() => {
    if (Platform.OS !== "web") return;
    if (!deepLinkPostId) {
      deepLinkAttemptsRef.current = 0;
      return;
    }
    // wait for the initial load to finish to avoid early "not found"
    if (loading) return;

    const idx = timelineItems.findIndex((it) => !isSlotItem(it) && it.id === deepLinkPostId);
    if (idx >= 0) {
      requestAnimationFrame(() => {
        listRef.current?.scrollToIndex({ index: idx, animated: false });
      });
      setDeepLinkPostId(null);
      deepLinkAttemptsRef.current = 0;
      return;
    }

    if (nextUrl && !loadingMore) {
      if (deepLinkAttemptsRef.current >= MAX_DEEP_LINK_ATTEMPTS) {
        setError(`Permalink not found (gave up paging): ${deepLinkPostId}`);
        setDeepLinkPostId(null);
        deepLinkAttemptsRef.current = 0;
        return;
      }
      deepLinkAttemptsRef.current += 1;
      void loadMore();
      return;
    }

    if (!nextUrl && !loadingMore) {
      setError(`Permalink not found: ${deepLinkPostId}`);
      setDeepLinkPostId(null);
      deepLinkAttemptsRef.current = 0;
    }
  }, [deepLinkPostId, loading, timelineItems, nextUrl, loadingMore, loadMore]);


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
          const enabled = process.env.EXPO_PUBLIC_USE_SLOT === "1";
          const banners = SLOT_BANNERS;

          if (enabled && banners.length) {
            const startIndex = hashString(item.id) % banners.length;

            return (
              <View style={{ paddingHorizontal: 16, paddingBottom: 12 }}>
                <View style={{ flexDirection: "row", alignItems: "flex-start" }}>
                  {/* keep alignment with the mascot column */}
                  <View style={{ flex: 1 }}>
                    <SlotCard banners={banners} startIndex={startIndex} variant="inline" />
                  </View>
                </View>
              </View>
            );
          }

          const open = () => {
            if (!item.url) return;
            void Linking.openURL(item.url);
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
        const url = buildPermalink(item.id, item.permalink);
        return (
          <View style={{ paddingHorizontal: 16, paddingBottom: 12 }}>
            <View style={{ flexDirection: "row", alignItems: "flex-start" }}>
              <View style={{ width: MASCOT_COL_W, alignItems: "center" }}>
                <View style={{ marginTop: 2 }}>
                  <Mascot />
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
                      {item.place ? <Text style={{ color: TEXT_DIM }}>• {item.place}</Text> : null}
                    </View>
                    
                    <FeedBubbleImage uris={imageUris} />
                    <Text style={{ color: "#000000", marginTop: 8, fontSize: 16, lineHeight: 22 }}>{item.text}</Text>

                    {Array.isArray(item.links) && item.links.length > 0 ? (
                      <View style={{ flexDirection: "row", flexWrap: "wrap", gap: 6, marginTop: 8 }}>
                        {item.links.slice(0, 5).map((u, idx) => (
                          <Pressable
                            key={`${u}-${idx}`}
                            onPress={() => {
                              try { void Linking.openURL(u); } catch {}
                            }}
                          >
                            <Text style={{ color: "#0B57D0", textDecorationLine: "underline", fontSize: 12 }}>
                              🔗 {u}
                            </Text>
                          </Pressable>
                        ))}
                      </View>
                    ) : null}

                    <Pressable
                      onPress={async () => {
                        // On web: copy permalink (shareable)
                        if (Platform.OS === "web" && typeof navigator !== "undefined" && navigator.clipboard?.writeText) {
                          try {
                            await navigator.clipboard.writeText(url);
                            return;
                          } catch {}
                        }
                        // Fallback: open
                        try {
                          await Linking.openURL(url);
                        } catch {}
                      }}
                    >
                      <Text style={{ opacity: 0.7, fontSize: 12 }}>🔗 Permalink</Text>
                    </Pressable>
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
      {/* Left sidebar: grows to consume extra space */}
      <View style={{ flex: 1, minWidth: SIDEBAR_W, minHeight: 0 }}>
        <Slot side="left" />
      </View>

      {/* Center: keep 760 as the “target” width, but allow shrinking */}
      <View style={{ width: CONTENT_MAX_W, minWidth: 0, flexShrink: 1 }}>
        {list}
      </View>

      {/* Right sidebar: grows to consume extra space */}
      <View style={{ flex: 1, minWidth: SIDEBAR_W, minHeight: 0 }}>
        <Slot side="right" />
      </View>
    </View>
  );
}