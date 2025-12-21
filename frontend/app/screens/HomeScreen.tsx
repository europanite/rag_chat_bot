import React, { useCallback, useEffect, useMemo, useState } from "react";
import {
  ActivityIndicator,
  FlatList,
  Linking,
  RefreshControl,
  Text,
  TouchableOpacity,
  View,
} from "react-native";

type FeedItem = {
  id: string;
  date: string; // YYYY-MM-DD
  text: string;
  place?: string;
};

type Feed = {
  updated_at?: string;
  place?: string;
  items: FeedItem[];
};

const CARD_BG = "#cccccc";
const TEXT_DIM = "#333333";

function safeJsonParse(raw: string): unknown | null {
  try {
    return JSON.parse(raw) as unknown;
  } catch {
    return null;
  }
}

function normalizeFeed(parsed: unknown): Feed | null {
  if (!parsed) return null;

  // Feed shape: { items: [...] }
  if (typeof parsed === "object" && !Array.isArray(parsed)) {
    const obj = parsed as any;

    // New shape: { items: [...] }
    if (Array.isArray(obj.items)) {
      const items: FeedItem[] = obj.items
        .map((it: any, idx: number): FeedItem | null => {
          const date = typeof it?.date === "string" ? it.date : "";
          const text = typeof it?.text === "string" ? it.text : "";
          const id =
            typeof it?.id === "string"
              ? it.id
              : typeof it?.generated_at === "string"
                ? it.generated_at
                : date || String(idx);
          if (!date || !text) return null;
          const place = typeof it?.place === "string" && it.place ? it.place : undefined;
          return { id, date, text, place };
        })
        .filter(Boolean) as FeedItem[];

      return {
        updated_at: typeof obj.updated_at === "string" ? obj.updated_at : undefined,
        place: typeof obj.place === "string" ? obj.place : undefined,
        items,
      };
    }

    // Latest entry shape: { date, text, ... } (optionally includes feed_file/feed_url pointers)
    const date = typeof obj.date === "string" ? obj.date : "";
    const text = typeof obj.text === "string" ? obj.text : "";
    if (date && text) {
      const id =
        typeof obj.id === "string"
          ? obj.id
          : typeof obj.generated_at === "string"
            ? obj.generated_at
            : date;
      const place = typeof obj.place === "string" && obj.place ? obj.place : undefined;
      return {
        updated_at: typeof obj.generated_at === "string" ? obj.generated_at : undefined,
        place,
        items: [{ id, date, text, place }],
      };
    }
  }

  // Legacy shape: [ {date, text, ...}, ... ]
  if (Array.isArray(parsed)) {
    const items: FeedItem[] = parsed
      .map((it: any, idx: number): FeedItem | null => {
        const date = typeof it?.date === "string" ? it.date : "";
        const text = typeof it?.text === "string" ? it.text : "";
        const id =
          typeof it?.id === "string"
            ? it.id
            : typeof it?.generated_at === "string"
              ? it.generated_at
              : date || String(idx);
        if (!date || !text) return null;
        const place = typeof it?.place === "string" && it.place ? it.place : undefined;
        return { id, date, text, place };
      })
      .filter(Boolean) as FeedItem[];

    // Try to preserve a couple of top-level fields
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

export default function HomeScreen() {
  const FEED_URL = process.env.EXPO_PUBLIC_FEED_URL || "./latest.json";

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

  const sortedItems = useMemo(() => {
    const items = feed?.items ?? [];
    return [...items].sort((a, b) => (a.date < b.date ? 1 : a.date > b.date ? -1 : 0));
  }, [feed]);

  const latest = sortedItems[0] ?? null;

  const [effectiveUrl, setEffectiveUrl] = useState<string>(RESOLVED_FEED_URL);

  useEffect(() => {
    setEffectiveUrl(RESOLVED_FEED_URL);
  }, [RESOLVED_FEED_URL]);

  const load = useCallback(async () => {
    let currentEffectiveUrl = RESOLVED_FEED_URL;

    try {
      setError(null);

      const fetchJson = async (url: string): Promise<{ raw: string; parsed: unknown }> => {
        const finalUrl = addCacheBuster(url);
        const res = await fetch(finalUrl, { headers: { "Cache-Control": "no-cache" } });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const raw = await res.text();
        const parsed = safeJsonParse(raw);
        return { raw, parsed };
      };

      // 1) Fetch "entry" JSON (default: ./latest.json)
      setEffectiveUrl(RESOLVED_FEED_URL);
      const first = await fetchJson(RESOLVED_FEED_URL);

      // 2) If it contains a pointer to the real feed file, follow it
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

  const openFeed = useCallback(() => {
    if (!effectiveUrl) return;
    void Linking.openURL(effectiveUrl);
  }, [effectiveUrl]);

  const Header = (
    <View style={{ padding: 16, gap: 10 }}>
      <View style={{ gap: 6 }}>
        <Text style={{ fontSize: 22, fontWeight: "800", color: "#000000ff" }}>Yokosuka Days</Text>
        {/* <Text style={{ color: TEXT_DIM, fontSize: 12 }}>Feed: {effectiveUrl}</Text>
        {effectiveUrl !== RESOLVED_FEED_URL ? (
          <Text style={{ color: TEXT_DIM, fontSize: 12 }}>Entry: {RESOLVED_FEED_URL}</Text>
        ) : null} */}
      </View>

      <View style={{ flexDirection: "row", gap: 10, flexWrap: "wrap", alignItems: "center" }}>
        <TouchableOpacity
          onPress={openFeed}
          style={{
            backgroundColor: "#fff",
            paddingVertical: 8,
            paddingHorizontal: 12,
            borderRadius: 10,
          }}
        >
          {/* <Text style={{ fontWeight: "800" }}>Open feed JSON</Text> */}
        </TouchableOpacity>

        {feed?.updated_at ? <Text style={{ color: TEXT_DIM }}>Updated: {feed.updated_at}</Text> : null}
      </View>

      {latest ? (
        <View
          style={{
            backgroundColor: CARD_BG,
            borderRadius: 14,
            padding: 14,
            borderWidth: 1,
            borderColor: "#000000",
          }}
        >
          <Text style={{ color: "#000000", fontSize: 16, fontWeight: "800" }}>Latest ({latest.date})</Text>
          {latest.place ? <Text style={{ color: TEXT_DIM, marginTop: 4 }}>{latest.place}</Text> : null}
          <Text style={{ color: "#000000", marginTop: 10, fontSize: 16, lineHeight: 22 }}>{latest.text}</Text>
        </View>
      ) : null}

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
      <View style={{ flex: 1, alignItems: "center", justifyContent: "center", padding: 16 }}>
        <ActivityIndicator />
        <Text style={{ marginTop: 10, color: TEXT_DIM }}>Loading…</Text>
      </View>
    );
  }

  return (
    <FlatList
      data={sortedItems}
      keyExtractor={(it) => it.id}
      ListHeaderComponent={Header}
      refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} />}
      ListEmptyComponent={
        <View style={{ padding: 16 }}>
          <Text style={{ color: TEXT_DIM }}>No posts yet.</Text>
        </View>
      }
    />
  );
}
