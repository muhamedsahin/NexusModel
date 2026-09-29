import type { Metadata } from "next";
import "@fontsource-variable/space-grotesk";
import "@fontsource-variable/inter";
import "@fontsource-variable/jetbrains-mono";
import "katex/dist/katex.min.css";
import "./globals.css";
export const metadata: Metadata = {
  title: {
    default: "NexusModel — Computation, made visible.",
    template: "%s · NexusModel",
  },
  description:
    "C++20 neural network layers. Explicit backward, SIMD acceleration, and source-backed documentation in English and Turkish.",
};
const themeScript = `try{var t=localStorage.getItem('nx-theme');document.documentElement.dataset.theme=t==='light'?'light':'dark'}catch(e){}`;
export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="tr" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeScript }} />
      </head>
      <body>{children}</body>
    </html>
  );
}
