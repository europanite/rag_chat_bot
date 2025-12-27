import React, { useCallback, useEffect, useMemo, useState } from "react";
import {
  ActivityIndicator,
  FlatList,
  Image,
  Linking,
  Platform,
  RefreshControl,
  Text,
  useWindowDimensions,
  View,
} from "react-native";

type FeedItem = {
  id: string;
  date: string; // YYYY-MM-DD
  text: string;
  place?: string;
  generated_at?: string; // ISO string (often Z)
};

type Feed = {
  updated_at?: string;
  place?: string;
  items: FeedItem[];
};

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
          return { id, date, text, place, generated_at };
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
      const updated_at = generated_at;
      return { updated_at, place, items: [{ id, date, text, place, generated_at }] };
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
        return { id, date, text, place, generated_at };
      })
      .filter(Boolean) as FeedItem[];

    const last = parsed.length > 0 ? (parsed[parsed.length - 1] as any) : null;
    const updated_at = typeof last?.generated_at === "string" ? last.generated_at : undefined;
    const place = typeof last?.place === "string" ? last.place : undefined;

    return { updated_at, place, items };
  }

  return null;
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
    if (/^(https?:)?\/\//i.test(envUri) || envUri.startsWith("data:")) return envUri;
    if (Platform.OS === "web" && typeof window !== "undefined") {
      try {
        return new URL(envUri, window.location.href).toString();
      } catch {
        return envUri;
      }
    }
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
        <Text style={{ color: "#ffffff", fontWeight: "900", fontSize: Math.max(18, Math.floor(size * 0.35)) }}>R</Text>
      </View>
    </Frame>
  );
}

function Slot() {
  const enabled = process.env.EXPO_PUBLIC_USE_SLOT === "1";
  if (!enabled) return null;

  return (
    <View
      style={{
        flex: 1,
        backgroundColor: APP_BG,
        borderWidth: 1,
        borderColor: BORDER,
        borderRadius: 12,
        padding: 12,
      }}
    >
      <Text style={{ color: TEXT_DIM, marginTop: 6, lineHeight: 18 }}>
      </Text>
    </View>
  );
}

export default function HomeScreen() {
  const FEED_URL = process.env.EXPO_PUBLIC_FEED_URL || "./latest.json";
  const { width } = useWindowDimensions();
  const showSidebars = width >= 980;

  const RESOLVED_FEED_URL = useMemo(() => {
    try {
      if (FEED_URL.startsWith("http://") || FEED_URL.startsWith("https://")) return FEED_URL;
      if (typeof window !== "undefined") return new URL(FEED_URL, window.location.href).toString();
    } catch {
      // ignore
    }
    return FEED_URL;
  }, [FEED_URL]);

  const [feed, setFeed] = useState<Feed | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [refreshing, setRefreshing] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const [nextUrl, setNextUrl] = useState<string | null>(null);
  const [loadingMore, setLoadingMore] = useState<boolean>(false);

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

  const [effectiveUrl, setEffectiveUrl] = useState<string>(RESOLVED_FEED_URL);

  useEffect(() => {
    setEffectiveUrl(RESOLVED_FEED_URL);
  }, [RESOLVED_FEED_URL]);

  const load = useCallback(async () => {
    let currentEffectiveUrl = RESOLVED_FEED_URL;

    try {
      setError(null);
      setNextUrl(null);


      setEffectiveUrl(RESOLVED_FEED_URL);
      const first = await fetchJson(RESOLVED_FEED_URL);

      const pointer = getFeedPointer(first.parsed);
      let target = first;

      if (pointer) {
        currentEffectiveUrl = resolveUrl(pointer, RESOLVED_FEED_URL);
        setEffectiveUrl(currentEffectiveUrl);
        target = await fetchJson(currentEffectiveUrl);
      }

      const normalized = normalizeFeed(target.parsed);
      if (!normalized) {
        const preview = target.raw.slice(0, 180).replace(/\s+/g, " ").trim();
        throw new Error(`Invalid feed JSON shape\nURL: ${currentEffectiveUrl}\nRAW: ${preview}`);
      }

      const nextPointer = getNextPointer(target.parsed);
      setNextUrl(nextPointer ? resolveUrl(nextPointer, currentEffectiveUrl) : null);

      setFeed(normalized);
    } catch (e: any) {
      setError(e?.message ?? "Failed to load feed");
      setFeed(null);
    } finally {
      setLoading(false);
    }
  }, [RESOLVED_FEED_URL]);

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
    <View style={{ padding: 16, gap: 10 }}>
      <View style={{ flexDirection: "row", gap: 10, flexWrap: "wrap", alignItems: "center" }}>
        {feed?.updated_at ? <Text style={{ color: TEXT_DIM }}>Updated At: {formatJst(feed.updated_at, true)}</Text> : null}
      </View>

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
      style={{ flex: 1, backgroundColor: APP_BG }}
      contentContainerStyle={{ paddingBottom: 18 }}
      data={sortedItems}
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
      renderItem={({ item }) => (
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
                    {item.generated_at ? <Text style={{ color: TEXT_DIM }}>{formatJst(item.generated_at)}</Text> : null}
                    {item.place ? <Text style={{ color: TEXT_DIM }}>• {item.place}</Text> : null}
                  </View>

                  <Text style={{ color: "#000000", marginTop: 8, fontSize: 16, lineHeight: 22 }}>{item.text}</Text>
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
      )}
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
    <View style={{ flex: 1, flexDirection: "row", justifyContent: "center", backgroundColor: APP_BG }}>
      <View style={{ width: SIDEBAR_W, paddingTop: 16, paddingLeft: 12, minHeight: 0 }}>
        <Slot />
      </View>

      <View style={{ flex: 1, maxWidth: CONTENT_MAX_W }}>{list}</View>

      <View style={{ width: SIDEBAR_W, paddingTop: 16, paddingRight: 12, minHeight: 0 }}>
        <Slot />
      </View>
    </View>
  );
}
