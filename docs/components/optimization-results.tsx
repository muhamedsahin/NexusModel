import Link from "next/link";
import data from "@/lib/optimization-results.json";
import { Locale } from "@/lib/types";

const labels: Record<string, string> = {
  linear_forward_128_512_512: "Linear · forward · 128 × 512 × 512",
  linear_backward_128_512_512: "Linear · backward · 128 × 512 × 512",
  softmax_128_512: "Softmax · 128 × 512",
  layernorm_128_512: "LayerNorm · 128 × 512",
};

export function OptimizationResults({ lang }: { lang: Locale }) {
  const t = (tr: string, en: string) => lang === "tr" ? tr : en;
  return (
    <section className="optimization-study" aria-label={t("Optimizasyon sonuçları", "Optimization results")}>
      <div className="results-heading">
        <div>
          <span className="eyebrow">NEXUSMODEL / BEFORE & AFTER</span>
          <h2>{t("Aynı çekirdek. Daha az süre.", "Same computation. Less time.")}</h2>
        </div>
        <span>{data.metadata.measured_at.slice(0, 10)} · 90 SAMPLES</span>
      </div>
      <p>{t(
        "Arşivlenmiş eski kaynak ve optimize sürüm aynı C++ ölçüm koduyla derlendi. Üç turda sürüm sırası değiştirilerek ölçüldü; değerler medyan süredir.",
        "Archived sources and the optimized revision were built with the same C++ timing harness. Three rounds alternated revision order; values are median times.",
      )}</p>
      <div className="optimization-grid">
        {data.records.map((row) => (
          <article className="optimization-card" key={row.operation}>
            <span>{labels[row.operation] ?? row.operation}</span>
            <strong>{row.speedup.toFixed(2)}<small>×</small></strong>
            <p>{row.before_ms.toFixed(4)} → <b>{row.after_ms.toFixed(4)} ms</b></p>
            <div className="optimization-track"><i style={{width: `${Math.min(100, row.after_ms / row.before_ms * 100)}%`}} /></div>
            <small>{t("Eski süreye göre yeni süre", "New time relative to old time")}</small>
          </article>
        ))}
      </div>
      <p className="subtle">{t(
        "Float32, tek hesaplama iş parçacığı. Backward tampon sıfırlamasını içerir. Güç profili, sıcaklık ve işlemci çekirdeği sabitlenmedi. Bunlar tüm modeller için genel hız garantisi değildir.",
        "Float32, one compute thread. Backward includes gradient buffer clearing. Power, temperature and CPU affinity were not fixed. These are not universal speed guarantees for all models.",
      )}</p>
      <div className="comparison-downloads">
        <a className="button secondary compact" href="/NexusModel-Kullanim-Kilavuzu.pdf" download>{t("Güncel kılavuz · PDF", "Updated guide · PDF")}</a>
        <a className="button secondary compact" href="/optimization-results.json" download>{t("Eski/yeni ham örnekler · JSON", "Before/after raw samples · JSON")}</a>
        <Link className="text-link" href={`/${lang}/docs/performance-engine/`}>{t("Hesap motorunun ayrıntıları →", "Compute engine details →")}</Link>
      </div>
    </section>
  );
}
