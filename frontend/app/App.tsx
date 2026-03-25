import React, { useEffect } from "react";
import { Platform, View } from "react-native";
import { NavigationContainer } from "@react-navigation/native";
import { createNativeStackNavigator } from "@react-navigation/native-stack";
import { SafeAreaProvider } from "react-native-safe-area-context";
import { StatusBar } from "expo-status-bar";

import { AuthProvider } from "./context/Auth";
import SettingsBar from "./components/SettingsBar";
import HomeScreen from "./screens/HomeScreen";

const APP_TITLE = "GOODDAY YOKOSUKA";
const GA_MEASUREMENT_ID = (process.env.EXPO_PUBLIC_GA_MEASUREMENT_ID ?? "").trim();

function installGoogleAnalytics() {
  if (Platform.OS !== "web" || !GA_MEASUREMENT_ID || typeof document === "undefined") {
    return;
  }

  const w = window as typeof window & {
    dataLayer?: unknown[];
    gtag?: (...args: unknown[]) => void;
  };

  if (!document.querySelector(`script[data-ga4-id="${GA_MEASUREMENT_ID}"]`)) {
    const script = document.createElement("script");
    script.async = true;
    script.src = `https://www.googletagmanager.com/gtag/js?id=${GA_MEASUREMENT_ID}`;
    script.setAttribute("data-ga4-id", GA_MEASUREMENT_ID);
    document.head.appendChild(script);
  }

  w.dataLayer = w.dataLayer || [];
  if (typeof w.gtag !== "function") {
    w.gtag = (...args: unknown[]) => {
      w.dataLayer!.push(args);
    };
  }

  w.gtag("js", new Date());
  w.gtag("config", GA_MEASUREMENT_ID, {
    page_path: `${window.location.pathname}${window.location.search}`,
    page_location: window.location.href,
    page_title: document.title,
    send_page_view: true,
  });
}

const Stack = createNativeStackNavigator();

export default function App() {
  useEffect(() => {
    installGoogleAnalytics();
  }, []);

  return (
    <SafeAreaProvider>
      <StatusBar style="dark" />
      <AuthProvider>
        <NavigationContainer>
          <SettingsBar title={APP_TITLE}/>
          <View style={{ flex: 1 }}>
            <Stack.Navigator screenOptions={{ headerShown: false }}>
              <Stack.Screen name={APP_TITLE} component={HomeScreen} />
            </Stack.Navigator>
          </View>
        </NavigationContainer>
      </AuthProvider>
    </SafeAreaProvider>
  );
}
