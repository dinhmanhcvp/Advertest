import { Inter, JetBrains_Mono } from "next/font/google";
import Script from "next/script";
import { Toaster } from "sonner";
import AppProviders from "@/components/AppProviders";
import AppShell from "@/components/layout/AppShell";
import "./globals.css";

const inter = Inter({ subsets: ["latin", "vietnamese"] });
const jetbrainsMono = JetBrains_Mono({ subsets: ["latin"], variable: "--font-jb-mono" });

export const metadata = {
  title: "AdverTest — Nền tảng đánh giá & phòng thủ AI đối kháng",
  description: "Nền tảng kiểm thử độ bền vững và huấn luyện phòng thủ mô hình AI thị giác chuyên sâu.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="vi" className={`${inter.className} ${jetbrainsMono.variable}`} suppressHydrationWarning>
      <head>
        <Script src="/runtime-config.js" strategy="beforeInteractive" />
      </head>
      <body className="antialiased bg-[var(--app-bg)] text-[var(--app-text)]">
        <AppProviders>
          <AppShell>{children}</AppShell>
        </AppProviders>
        <Toaster theme="dark" position="bottom-right" richColors />
      </body>
    </html>
  );
}
