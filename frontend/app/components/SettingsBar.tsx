import React from "react";
import { View, Text, TouchableOpacity, useWindowDimensions, Linking, Alert } from "react-native";

const REPO_URL = "https://github.com/europanite/rag_chat_bot";

// Public feedback URL (Google Form etc.)
// - Build-time env (GitHub Pages / Expo export): EXPO_PUBLIC_FEEDBACK_FORM_URL
// - Fallback: GitHub Issues (works even when env isn't set)
const RAW_CONTACT_URL = (process.env.EXPO_PUBLIC_FEEDBACK_FORM_URL ?? "").trim();
const CONTACT_URL =
  RAW_CONTACT_URL.startsWith("http://") || RAW_CONTACT_URL.startsWith("https://")
    ? RAW_CONTACT_URL
    : `${REPO_URL}/issues/new`;

function Btn({ title, onPress }: { title: string; onPress: () => void }) {
  return (
    <TouchableOpacity
      onPress={onPress}
      style={{ paddingVertical: 6, paddingHorizontal: 10, borderWidth: 1, borderRadius: 8, backgroundColor: "#fff" }}
      accessibilityRole="button"
      accessibilityLabel={title}
    >
      <Text style={{ fontWeight: "600" }}>{title}</Text>
    </TouchableOpacity>
  );
}

async function openUrl(url: string) {
  try {
    const ok = await Linking.canOpenURL(url);
    if (!ok) throw new Error("canOpenURL returned false");
    await Linking.openURL(url);
  } catch {
    Alert.alert("Open link failed", `Could not open:\n${url}`);
  }
}

export default function SettingsBar() {
  const { width } = useWindowDimensions();
  const isNarrow = width < 520;

  return (
    <View
      style={{
        paddingHorizontal: 12,
        paddingVertical: 10,
        borderBottomWidth: 1,
        backgroundColor: "#333366",
      }}
    >
      <View
        style={{
          flexDirection: isNarrow ? "column" : "row",
          gap: 10,
          alignItems: isNarrow ? "stretch" : "center",
          justifyContent: "space-between",
        }}
      >
        <TouchableOpacity onPress={() => openUrl(REPO_URL)} accessibilityRole="link">
          <Text style={{ fontSize: 8, color: "#fff" }}>
            Powered by <Text style={{ textDecorationLine: "underline", fontWeight: "700" }}>RAG Chat Bot</Text>
          </Text>
        </TouchableOpacity>

        <View
          style={{
            flexDirection: "row",
            gap: 8,
            flexWrap: "wrap",
            justifyContent: isNarrow ? "flex-start" : "flex-end",
          }}
        >
          <Btn title="Contact" onPress={() => openUrl(CONTACT_URL)} />
        </View>
      </View>
    </View>
  );
}
