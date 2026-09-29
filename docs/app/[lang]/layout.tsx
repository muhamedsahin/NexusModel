import { notFound } from "next/navigation";
import { Shell } from "@/components/shell";
import type { Locale } from "@/lib/types";
export function generateStaticParams() {
  return [{ lang: "tr" }, { lang: "en" }];
}
export const dynamicParams = false;
export default async function Layout({
  children,
  params,
}: {
  children: React.ReactNode;
  params: Promise<{ lang: string }>;
}) {
  const { lang } = await params;
  if (lang !== "tr" && lang !== "en") notFound();
  return <Shell lang={lang as Locale}>{children}</Shell>;
}
