import React from "react";
import { View, Text, TouchableOpacity, useWindowDimensions, Linking, Alert } from "react-native";

const REPO_URL = "https://github.com/europanite/rag_chat_bot";
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

type Props = { title?: string };

export default function SettingsBar({ title = "GOODDAY YOKOSUKA" }: Props) {
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
      {isNarrow ? (
        // NARROW: stack (avoid overlap)
        <View style={{ 
          gap: 8, 
          padding: 6,
          alignItems: "center"
        }}>
          <Text style={{ 
            fontSize: 32, 
            fontWeight: "512", 
            color: "#fff"
            }}>
              {title}
          </Text>
          <Text style={{ 
            fontWeight: "12",
            color: "#fff"
            }}
          >
            provides news for local and visitors.
          </Text>

          <View
            style={{
              flexDirection: "row",
              gap: 8,
              flexWrap: "wrap",
              justifyContent: "center",
            }}
          >
            <Btn title="Contact" onPress={() => openUrl(CONTACT_URL)} />
          </View>

          {/* Powered by: small + low emphasis */}
          <TouchableOpacity
            onPress={() => openUrl(REPO_URL)}
            accessibilityRole="link"
            style={{ opacity: 0.6 }}
          >
            <Text style={{ fontSize: 9, color: "#fff" }}>
              Powered by <Text style={{ fontWeight: "600" }}>RAG Chat Bot</Text>
            </Text>
          </TouchableOpacity>
        </View>
      ) : (
        // WIDE: title pinned to center (independent of button width)
        <View style={{ position: "relative", justifyContent: "center", minHeight: 28 }}>
          {/* Row content (left small label + right buttons) */}
          <View style={{ flexDirection: "row", alignItems: "center", justifyContent: "space-between" }}>
            {/* Powered by: unobtrusive (left) */}
            <TouchableOpacity
              onPress={() => openUrl(REPO_URL)}
              accessibilityRole="link"
              style={{ opacity: 0.55 }}
            >
              <Text style={{ fontSize: 9, color: "#fff" }}>
                Powered by <Text style={{ fontWeight: "600" }}>RAG Chat Bot</Text>
              </Text>
            </TouchableOpacity>

            {/* Buttons (right) */}
            <View style={{ flexDirection: "row", gap: 8, flexWrap: "wrap", justifyContent: "flex-end" }}>
              <Btn title="Contact" onPress={() => openUrl(CONTACT_URL)} />
            </View>
          </View>

          {/* Center title (overlay) */}
          <View
            pointerEvents="none"
            style={{
              position: "absolute",
              left: 0,
              right: 0,
              top: 0,
              bottom: 0,
              padding: 6,
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            <Text style={{ 
              fontSize: 32, 
              fontWeight: "512", 
              color: 
              "#fff" 
            }}>{title}</Text>
            <Text style={{ 
              fontWeight: "12",
              color: "#fff"
              }}
            >
              provides news for local and visitors.
            </Text>
          </View>
        </View>
      )}
    </View>
  );
}
