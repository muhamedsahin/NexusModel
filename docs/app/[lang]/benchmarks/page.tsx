import { Benchmarks } from "@/components/benchmarks";
import type { Locale } from "@/lib/types";
export const metadata = { title: "Benchmark · Performance Lab" };
export default async function Page({
  params,
}: {
  params: Promise<{ lang: Locale }>;
}) {
  const { lang } = await params;
  return <Benchmarks lang={lang} />;
}
