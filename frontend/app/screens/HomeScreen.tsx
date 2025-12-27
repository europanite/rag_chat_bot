import React, { useCallback, useEffect, useMemo, useState } from "react";
import {
  ActivityIndicator,
  FlatList,
  Image,
  Platform,
  Pressable,
  RefreshControl,
  SafeAreaView,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from "react-native";

const BG = "#efeff6";
const NAVY = "#383960";
const BORDER = "#2b2b2b";

type FeedItem = {
  id: string;
  date: string; // YYYY-MM-DD
  text: string;
  place?: string;
  generated_at?: string;
  image?: string; // optional (absolute URL or /public path)
  image_prompt?: string; // optional (for matching)
};

type Feed = {
  updated_at?: string;
  place?: string;
  items: FeedItem[];
};

type Latest = {
  date?: string;
  place?: string;
  text?: string;
  generated_at?: string;
  image_url?: string;
  image_prompt?: string;
  image_model?: string;
  image_generated_at?: string;
};

function tryJsonParse(s: string): any | null {
  try {
    return JSON.parse(s);
  } catch {
    return null;
  }
}

function parseFeedShape(raw: string): Feed | null {
  const parsed = tryJsonParse(raw);
  if (!parsed) return null;

  // Shape A: { items: [...] }
  if (typeof parsed === "object" && parsed && !Array.isArray(parsed)) {
    const obj: any = parsed;

    if (Array.isArray(obj.items)) {
      const items: FeedItem[] = obj.items
        .map((it: any, idx: number): FeedItem | null => {
          const date = typeof it?.date === "string" ? it.date : "";
          const text = typeof it?.text === "string" ? it.text : "";
          if (!date || !text) return null;

          const id =
            typeof it?.id === "string"
              ? it.id
              : typeof it?.generated_at === "string"
              ? it.generated_at
              : `${date}-${idx}`;

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

          return { id, date, text, place, generated_at, image, image_prompt };
        })
        .filter(Boolean) as FeedItem[];

      return {
        updated_at: typeof obj.updated_at === "string" ? obj.updated_at : undefined,
        place: typeof obj.place === "string" ? obj.place : undefined,
        items,
      };
    }

    // Shape B: latest.json single object { date, text, ... }
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

      return { updated_at, place, items: [{ id, date, text, place, generated_at, image, image_prompt }] };
    }
  }

  // Shape C: plain array of items
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

        return { id, date, text, place, generated_at, image, image_prompt };
      })
      .filter(Boolean) as FeedItem[];

    const last = parsed.length > 0 ? (parsed[parsed.length - 1] as any) : null;
    const updated_at = typeof last?.generated_at === "string" ? last.generated_at : undefined;
    const place = typeof last?.place === "string" ? last.place : undefined;

    return { updated_at, place, items };
  }

  return null;
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
  const s = String(p ?? "").trim();
  if (!s) return "";
  if (/^(https?:)?\/\//i.test(s) || s.startsWith("data:")) return s;

  // GitHub Pages repo subpath safety: "/share_sd/..." should be treated as "./share_sd/..."
  if (Platform.OS === "web" && s.startsWith("/")) return `.${s}`;
  return s;
}

function buildSharePrompt(text: string, place?: string): string {
  const t = String(text ?? "").replace(/\s+/g, " ").trim().slice(0, 240);
  const p = String(place ?? "").trim();
  return p ? `cinematic illustration, ${p}, based on this short story: ${t}` : `cinematic illustration, based on this short story: ${t}`;
}

function resolveUrl(p: string, assetBase: string): string {
  const s = String(p ?? "").trim();
  if (!s) return "";
  if (/^(https?:)?\/\//i.test(s) || s.startsWith("data:")) return s;

  const base = String(assetBase ?? "").trim();
  if (!base) return s;

  // assetBase is expected to end with "/" or be a dir; keep it simple
  if (s.startsWith("./")) return base.replace(/\/+$/, "/") + s.slice(2);
  if (s.startsWith("/")) return base.replace(/\/+$/, "") + s;
  return base.replace(/\/+$/, "/") + s;
}

const FeedBubbleImage: React.FC<{ uris?: string[] }> = ({ uris }) => {
  const [idx, setIdx] = useState(0);
  const [hidden, setHidden] = useState(false);

  // Reset when the candidate list changes (e.g., new feed item).
  useEffect(() => {
    setIdx(0);
    setHidden(false);
  }, [uris?.join("|")]);

  const uri = (uris ?? [])[idx] ?? "";

  // If there's no image or all candidates failed to load (404 etc.), render nothing (text only).
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
        style={{ width: "100%", aspectRatio: 16 / 9 }}
        resizeMode="cover"
        accessibilityLabel="Generated image"
        onError={() => {
          if (uris && idx + 1 < uris.length) {
            setIdx(idx + 1);
          } else {
            setHidden(true);
          }
        }}
      />
    </View>
  );
};

export default function HomeScreen() {
  const [feed, setFeed] = useState<Feed | null>(null);
  const [shareSd, setShareSd] = useState<ShareSdIndex | null>(null);
  const [loading, setLoading] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Compute an asset base for GitHub Pages if possible.
  const assetBase = useMemo(() => {
    if (Platform.OS !== "web") return "";
    // Prefer <base href="..."> if present.
    const baseEl = typeof document !== "undefined" ? document.querySelector("base") : null;
    const baseHref = baseEl?.getAttribute("href") || "";
    if (baseHref) return baseHref;

    // Fallback: use current path dir as base.
    try {
      const u = new URL(window.location.href);
      const dir = u.pathname.replace(/[^/]*$/, ""); // strip filename
      return `${u.origin}${dir}`;
    } catch {
      return "";
    }
  }, []);

  const fetchText = useCallback(async (url: string) => {
    const res = await fetch(url, { cache: "no-store" as any });
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
    return await res.text();
  }, []);

  const loadAll = useCallback(async () => {
    setError(null);
    setLoading(true);
    try {
      // 1) feed/latest
      const feedCandidates = [
        "/feed/latest.json",
        "/feed/output.json",
        "/feed/feed.json",
        "/latest.json",
        "/feed/latest.md",
      ].map((p) => resolveUrl(normalizeWebAssetPath(p), assetBase));

      let loadedFeed: Feed | null = null;
      for (const u of feedCandidates) {
        try {
          const t = await fetchText(u);
          loadedFeed = parseFeedShape(t);
          if (loadedFeed?.items?.length) break;
        } catch {
          // ignore
        }
      }

      // 2) share_sd index (optional)
      const shareCandidates = ["/share_sd/index.json"].map((p) => resolveUrl(normalizeWebAssetPath(p), assetBase));
      let loadedShare: ShareSdIndex | null = null;
      for (const u of shareCandidates) {
        try {
          const t = await fetchText(u);
          const parsed = tryJsonParse(t);
          if (parsed && typeof parsed === "object" && Array.isArray((parsed as any).items)) {
            loadedShare = parsed as ShareSdIndex;
            break;
          }
        } catch {
          // ignore
        }
      }

      setFeed(loadedFeed);
      setShareSd(loadedShare);
    } catch (e: any) {
      setError(e?.message || String(e));
    } finally {
      setLoading(false);
    }
  }, [assetBase, fetchText]);

  useEffect(() => {
    loadAll();
  }, [loadAll]);

  const onRefresh = useCallback(async () => {
    setRefreshing(true);
    try {
      await loadAll();
    } finally {
      setRefreshing(false);
    }
  }, [loadAll]);

  const sharePromptToImage = useMemo(() => {
    const map = new Map<string, string>();
    if (!shareSd?.items?.length) return map;

    for (const it of shareSd.items) {
      if (!it?.image) continue;
      const img = resolveUrl(normalizeWebAssetPath(it.image), assetBase);
      if (it.prompt) map.set(it.prompt, img);
      if (it.date && it.place) map.set(`${it.date}|${it.place}`, img);
    }
    return map;
  }, [assetBase, shareSd]);

  const getImageUrisForItem = useCallback(
    (item: FeedItem): string[] => {
      const out: string[] = [];

      const push = (maybePath: string) => {
        const s = String(maybePath ?? "").trim();
        if (!s) return;
        const resolved = resolveUrl(normalizeWebAssetPath(s), assetBase);
        if (!out.includes(resolved)) out.push(resolved);
      };

      // 1) If the feed already contains an explicit image path, use it first.
      if (item.image) push(item.image);

      // 2) Deterministic: try "/image/<item.id>.png" (feed/image stem match).
      const idStem = String(item.id ?? "").trim();
      if (idStem) {
        const lower = idStem.toLowerCase();
        if (lower.endsWith(".png") || lower.endsWith(".jpg") || lower.endsWith(".jpeg") || lower.endsWith(".webp")) {
          push(`/image/${encodeURIComponent(idStem)}`);
        } else {
          push(`/image/${encodeURIComponent(idStem)}.png`);
        }
      }

      // 3) Fallback: try matching the Share SD index by prompt (deterministic) or by date+place.
      const place = item.place || feed?.place;
      const prompt = item.image_prompt || buildSharePrompt(item.text, place);

      const fromPrompt = sharePromptToImage.get(prompt);
      if (fromPrompt) push(fromPrompt);

      if (item.date && place) {
        const byKey = sharePromptToImage.get(`${item.date}|${place}`);
        if (byKey) push(byKey);
      }

      return out;
    },
    [assetBase, feed?.place, sharePromptToImage],
  );

  const items = feed?.items ?? [];

  const renderItem = useCallback(
    ({ item }: { item: FeedItem }) => {
      const when = item.generated_at || item.date;
      const where = item.place || feed?.place;
      const imageUris = getImageUrisForItem(item);

      return (
        <View style={{ flexDirection: "row", gap: 16, marginBottom: 18 }}>
          <View style={{ width: 92, alignItems: "center" }}>
            <Image
              source={{ uri: resolveUrl(normalizeWebAssetPath("/images/avatar.png"), assetBase) }}
              style={{ width: 72, height: 72, borderRadius: 18, borderWidth: 2, borderColor: BORDER, backgroundColor: "#fff" }}
              onError={() => {}}
            />
          </View>

          <View style={{ flex: 1 }}>
            <View style={styles.bubble}>
              <Text style={styles.meta}>
                {when}
                {where ? `  ・  ${where}` : ""}
              </Text>

              {/* IMAGE: show above text inside the bubble, only if it exists */}
              <FeedBubbleImage uris={imageUris} />

              <Text style={styles.body}>{item.text}</Text>
            </View>
          </View>
        </View>
      );
    },
    [assetBase, feed?.place, getImageUrisForItem],
  );

  return (
    <SafeAreaView style={{ flex: 1, backgroundColor: BG }}>
      <View style={styles.header}>
        <Text style={styles.title}>GOODDAY YOKOSUKA</Text>
        <View style={styles.contactBtn}>
          <Text style={{ fontWeight: "700" }}>Contact</Text>
        </View>
      </View>

      <View style={styles.container}>
        <View style={styles.side} />
        <View style={styles.main}>
          <View style={{ paddingHorizontal: 18, paddingTop: 10, paddingBottom: 6 }}>
            <Text style={{ color: "#111", fontWeight: "600" }}>
              Updated At: {feed?.updated_at ? feed.updated_at : "—"}
            </Text>
          </View>

          {error ? (
            <View style={{ padding: 18 }}>
              <Text style={{ color: "#c00", fontWeight: "700" }}>Error</Text>
              <Text style={{ color: "#c00", marginTop: 6 }}>{error}</Text>
            </View>
          ) : null}

          {loading && !items.length ? (
            <View style={{ paddingTop: 40 }}>
              <ActivityIndicator />
            </View>
          ) : (
            <FlatList
              data={items}
              keyExtractor={(it) => it.id}
              renderItem={renderItem}
              contentContainerStyle={{ padding: 18 }}
              refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} />}
            />
          )}
        </View>
        <View style={styles.side} />
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  header: {
    height: 62,
    backgroundColor: NAVY,
    justifyContent: "center",
    alignItems: "center",
    paddingHorizontal: 16,
  },
  title: {
    color: "white",
    fontSize: 30,
    fontWeight: "800",
    letterSpacing: 1.5,
  },
  contactBtn: {
    position: "absolute",
    right: 16,
    top: 16,
    backgroundColor: "#fff",
    borderWidth: 1,
    borderColor: BORDER,
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 12,
  },
  container: {
    flex: 1,
    flexDirection: "row",
  },
  side: {
    width: 260,
    backgroundColor: BG,
  },
  main: {
    flex: 1,
    backgroundColor: BG,
    borderLeftWidth: 1,
    borderRightWidth: 1,
    borderColor: "#d0d0da",
  },
  bubble: {
    backgroundColor: "#fff",
    borderWidth: 2,
    borderColor: BORDER,
    borderRadius: 14,
    paddingVertical: 12,
    paddingHorizontal: 14,
  },
  meta: {
    fontSize: 13,
    fontWeight: "700",
    color: "#111",
    marginBottom: 6,
  },
  body: {
    fontSize: 18,
    color: "#111",
    lineHeight: 26,
  },
});
