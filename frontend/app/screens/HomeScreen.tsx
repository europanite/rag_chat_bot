import React, { useEffect, useState } from "react";
import {
  View,
  Text,
  TextInput,
  Button,
  ScrollView,
  Platform,
  KeyboardAvoidingView,
} from "react-native";

const API_BASE = process.env.EXPO_PUBLIC_API_BASE!;

type RagChunk = {
  id?: string | null;
  text: string;
  distance?: number | null;
  metadata?: Record<string, any>;
};

type RagQueryResponse = {
  answer: string;
  context: string[];
  chunks?: RagChunk[];
};

type RagStatusResponse = {
  docs_dir: string;
  json_files: number;
  chunks_in_store: number;
  files: string[];
};

type RagReindexResponse = {
  documents: number;
  chunks: number;
  files: number;
};

export default function HomeScreen() {
  const [status, setStatus] = useState<RagStatusResponse | null>(null);
  const [statusLoading, setStatusLoading] = useState(false);
  const [reindexLoading, setReindexLoading] = useState(false);
  const [reindexResult, setReindexResult] = useState<RagReindexResponse | null>(null);

  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [chunks, setChunks] = useState<RagChunk[]>([]);
  const [queryLoading, setQueryLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchStatus = async () => {
    setStatusLoading(true);
    try {
      const res = await fetch(`${API_BASE}/rag/status`);
      if (!res.ok) {
        const body = await res.text();
        throw new Error(`Status failed: ${res.status} ${body}`);
      }
      const data: RagStatusResponse = await res.json();
      setStatus(data);
    } catch (e: any) {
      console.error(e);
      setError(e.message ?? String(e));
    } finally {
      setStatusLoading(false);
    }
  };

  useEffect(() => {
    void fetchStatus();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleReindex = async () => {
    setReindexLoading(true);
    setError(null);
    setReindexResult(null);

    try {
      const res = await fetch(`${API_BASE}/rag/reindex`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      });

      if (!res.ok) {
        const body = await res.text();
        throw new Error(`Reindex failed: ${res.status} ${body}`);
      }

      const data: RagReindexResponse = await res.json();
      setReindexResult(data);
      await fetchStatus();
    } catch (e: any) {
      console.error(e);
      setError(e.message ?? String(e));
    } finally {
      setReindexLoading(false);
    }
  };

  const handleQuery = async () => {
    if (!question.trim()) {
      setError("Please enter a question.");
      return;
    }

    setQueryLoading(true);
    setError(null);
    setAnswer("");
    setChunks([]);

    try {
      const res = await fetch(`${API_BASE}/rag/query`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          question,
          top_k: 5,
        }),
      });

      if (!res.ok) {
        const body = await res.text();
        throw new Error(`Query failed: ${res.status} ${body}`);
      }

      const data: RagQueryResponse = await res.json();
      setAnswer(data.answer);

      // Prefer structured chunks if provided; fallback to `context` strings.
      if (Array.isArray(data.chunks) && data.chunks.length > 0) {
        setChunks(data.chunks);
      } else {
        setChunks((data.context ?? []).map((t) => ({ text: t })));
      }
    } catch (e: any) {
      console.error(e);
      setError(e.message ?? String(e));
    } finally {
      setQueryLoading(false);
    }
  };

  const handleClear = () => {
    setQuestion("");
    setAnswer("");
    setChunks([]);
    setError(null);
    setReindexResult(null);
  };

  const canClear = Boolean(question || answer || chunks.length > 0 || error || reindexResult);

  return (
    <KeyboardAvoidingView
      style={{ flex: 1, backgroundColor: "#f5f5f5" }}
      behavior={Platform.OS === "ios" ? "padding" : undefined}
    >
      <ScrollView
        contentContainerStyle={{
          paddingHorizontal: 16,
          paddingVertical: 20,
          alignItems: "center",
        }}
        keyboardShouldPersistTaps="handled"
      >
        <View style={{ width: "100%", maxWidth: 720, gap: 24 }}>
          {/* Section 1: Index / Status */}
          <View
            style={{
              backgroundColor: "#fff",
              padding: 16,
              borderRadius: 10,
              borderWidth: 1,
              borderColor: "#e0e0e0",
            }}
          >
            <Text style={{ fontSize: 18, fontWeight: "700", marginBottom: 8 }}>
              RAG Index (JSON directory)
            </Text>

            <Text style={{ color: "#555", marginBottom: 10 }}>
              Documents are loaded from JSON files (mounted from{" "}
              <Text style={{ fontWeight: "700" }}>./data/docs</Text>).
            </Text>

            <View style={{ marginBottom: 12 }}>
              <Button
                title={statusLoading ? "Refreshing..." : "Refresh status"}
                onPress={fetchStatus}
                disabled={statusLoading}
              />
            </View>

            <View style={{ marginBottom: 12 }}>
              <Button
                title={reindexLoading ? "Reindexing..." : "Reindex from JSON files"}
                onPress={handleReindex}
                disabled={reindexLoading}
              />
            </View>

            {status && (
              <View style={{ gap: 6 }}>
                <Text>
                  <Text style={{ fontWeight: "700" }}>DOCS_DIR:</Text> {status.docs_dir}
                </Text>
                <Text>
                  <Text style={{ fontWeight: "700" }}>JSON files:</Text> {status.json_files}
                </Text>
                <Text>
                  <Text style={{ fontWeight: "700" }}>Chunks in store:</Text> {status.chunks_in_store}
                </Text>

                {status.files?.length > 0 && (
                  <View style={{ marginTop: 6 }}>
                    <Text style={{ fontWeight: "700", marginBottom: 4 }}>Files</Text>
                    {status.files.map((f) => (
                      <Text key={f} style={{ color: "#555" }}>
                        • {f}
                      </Text>
                    ))}
                  </View>
                )}
              </View>
            )}

            {reindexResult && (
              <View style={{ marginTop: 10 }}>
                <Text style={{ fontWeight: "700" }}>Reindex result</Text>
                <Text style={{ color: "#555" }}>
                  files={reindexResult.files}, documents={reindexResult.documents}, chunks={reindexResult.chunks}
                </Text>
              </View>
            )}
          </View>

          {/* Section 2: Ask */}
          <View
            style={{
              backgroundColor: "#fff",
              padding: 16,
              borderRadius: 10,
              borderWidth: 1,
              borderColor: "#e0e0e0",
            }}
          >
            <Text style={{ fontSize: 18, fontWeight: "700", marginBottom: 8 }}>
              Ask a question
            </Text>

            <Text style={{ fontWeight: "600", marginBottom: 4 }}>Question</Text>
            <TextInput
              value={question}
              onChangeText={setQuestion}
              placeholder=""
              style={{
                borderWidth: 1,
                borderColor: "#ccc",
                borderRadius: 6,
                paddingHorizontal: 8,
                paddingVertical: 8,
                backgroundColor: "#fff",
                marginBottom: 10,
              }}
            />

            <View style={{ gap: 8 }}>
              <Button
                title={queryLoading ? "Thinking..." : "Ask"}
                onPress={handleQuery}
                disabled={queryLoading}
              />
              <Button title="Clear" onPress={handleClear} disabled={!canClear} />
            </View>

            {error && (
              <Text style={{ marginTop: 10, color: "crimson" }}>Error: {error}</Text>
            )}
          </View>

          {/* Section 3: Answer */}
          <View
            style={{
              backgroundColor: "#fff",
              padding: 16,
              borderRadius: 10,
              borderWidth: 1,
              borderColor: "#e0e0e0",
            }}
          >
            <Text style={{ fontSize: 18, fontWeight: "700", marginBottom: 8 }}>
              Answer
            </Text>

            {answer ? (
              <Text style={{ lineHeight: 20 }}>{answer}</Text>
            ) : (
              <Text style={{ color: "#777" }}>
                The answer will appear here after you ask a question.
              </Text>
            )}

            {chunks.length > 0 && (
              <View style={{ marginTop: 14 }}>
                <Text style={{ fontWeight: "700", marginBottom: 6 }}>
                  Retrieved context
                </Text>

                {chunks.map((c, i) => {
                  const meta = c.metadata ?? {};
                  const source = meta.source ? String(meta.source) : "";
                  const file = meta.file ? String(meta.file) : "";
                  const docId = meta.doc_id ? String(meta.doc_id) : "";
                  const labelParts = [
                    c.id ? `id=${c.id}` : null,
                    docId ? `doc_id=${docId}` : null,
                    file ? `file=${file}` : null,
                    source ? `source=${source}` : null,
                  ].filter(Boolean);

                  return (
                    <View
                      key={`${c.id ?? i}`}
                      style={{
                        borderWidth: 1,
                        borderColor: "#eee",
                        borderRadius: 8,
                        padding: 10,
                        marginBottom: 10,
                        backgroundColor: "#fafafa",
                      }}
                    >
                      {labelParts.length > 0 && (
                        <Text style={{ color: "#666", marginBottom: 6 }}>
                          {labelParts.join(" • ")}
                        </Text>
                      )}
                      <Text style={{ lineHeight: 18 }}>{c.text}</Text>
                      {typeof c.distance === "number" && (
                        <Text style={{ color: "#888", marginTop: 6 }}>
                          distance={c.distance.toFixed(4)}
                        </Text>
                      )}
                    </View>
                  );
                })}
              </View>
            )}
          </View>
        </View>
      </ScrollView>
    </KeyboardAvoidingView>
  );
}
