"use client";
import { useState } from "react";
import { Download, CheckCircle2, Info, Code2 } from "lucide-react";
import Link from "next/link";
import data from "@/lib/comparison-results.json";
import { Locale } from "@/lib/types";
import { OptimizationResults } from "./optimization-results";
export function LibraryComparison({ lang }: { lang: Locale }) {
  const [id, setId] = useState(data.records[0].id);
  const [metric, setMetric] = useState<"median_ms" | "p95_ms">("median_ms");
  const t = (tr: string, en: string) => (lang === "tr" ? tr : en);
  const row = data.records.find((r) => r.id === id)!;
  const max = Math.max(...row.results.map((r) => r[metric]));
  const sorted = [...row.results].sort((a, b) => a[metric] - b[metric]);
  const best = sorted[0];
  return (
    <div className="library-comparison">
      <div className="comparison-intro">
        <div>
          <span className="eyebrow">LOCAL CPU STUDY / 3 LIBRARIES</span>
          <h2>
            NexusModel <span>vs</span> NumPy <span>vs</span> PyTorch
          </h2>
          <p>
            {t(
              "Aynı girdiler. Aynı işlem. Doğrulanmış çıktılar.",
              "Same inputs. Same operation. Verified outputs.",
            )}
          </p>
        </div>
        <div className="verification-badge">
          <CheckCircle2 size={17} />
          {t("Tüm çıktılar doğrulandı", "All outputs verified")}
        </div>
      </div>
      <div className="comparison-conditions">
        <span>{data.metadata.cpu}</span>
        <span>FLOAT32 / CPU</span>
        <span>{t("1 HESAPLAMA THREAD’İ", "1 COMPUTE THREAD")}</span>
        <span>{data.metadata.warmup} WARMUP · {data.metadata.samples} × {data.metadata.iterations_per_sample}</span>
        <span>{data.metadata.measured_at.slice(0, 10)}</span>
      </div>
      <div className="library-controls">
        <label>
          {t("İş yükü", "Workload")}
          <select
            aria-label={t("Karşılaştırma iş yükü", "Comparison workload")}
            value={id}
            onChange={(e) => setId(e.target.value)}
          >
            {data.records.map((r) => (
              <option value={r.id} key={r.id}>
                {r.operation} · {r.shape}
              </option>
            ))}
          </select>
        </label>
        <div className="metric-toggle">
          <button
            aria-pressed={metric === "median_ms"}
            onClick={() => setMetric("median_ms")}
          >
            {t("Medyan", "Median")}
          </button>
          <button
            aria-pressed={metric === "p95_ms"}
            onClick={() => setMetric("p95_ms")}
          >
            p95
          </button>
        </div>
      </div>
      <div className="library-chart">
        <div className="chart-top">
          <span>{row.definition}</span>
          <span>
            {t("SÜRE / ms · DÜŞÜK DAHA İYİ", "TIME / ms · LOWER IS BETTER")}
          </span>
        </div>
        {row.results.map((r, i) => (
          <div className="library-bar-row" key={r.library}>
            <div className="library-bar-label">
              <strong>
                <span className={`library-dot library-${i}`} />
                {r.library}
                <small>{r.version}</small>
              </strong>
              <span>
                {r[metric].toFixed(4)} <small>ms</small>
              </span>
            </div>
            <div className="library-track">
              <i
                className={`library-${i}`}
                style={{ width: `${(r[metric] / max) * 100}%` }}
              />
            </div>
          </div>
        ))}
        <div className="library-finding">
          <span>{t("BU İŞ YÜKÜNDE", "FOR THIS WORKLOAD")}</span>
          <p>
            <strong>{best.library}</strong>{" "}
            {t("en düşük süreyi ölçtü.", "recorded the lowest time.")}{" "}
            <b>{best[metric].toFixed(4)} ms</b>
          </p>
          <small>
            {t(
              "Sonuç sıralaması işlem, boyut, thread sayısı ve donanımla değişir.",
              "Rankings change with operation, size, thread count, and hardware.",
            )}
          </small>
        </div>
      </div>
      <div className="comparison-downloads">
        <a
          className="button secondary compact"
          href="/benchmark-comparison.csv"
          download
        >
          <Download size={14} />
          {t("Tüm veriler · CSV", "All data · CSV")}
        </a>
        <a
          className="button secondary compact"
          href="/benchmark-comparison.json"
          download
        >
          <Download size={14} />
          {t("Ham örnekler · JSON", "Raw samples · JSON")}
        </a>
        <Link className="text-link" href={`/${lang}/docs/library-comparison/`}>
          <Code2 size={15} />
          {t("Nasıl ölçüldü?", "How was it measured?")}
        </Link>
      </div>
      <div className="results-heading">
        <h2>{t("Bütün kütüphane ölçümleri", "All library measurements")}</h2>
        <span>
          {data.records.reduce((count, row) => count + row.results.length, 0)} {t("SONUÇ", "RESULTS")}
        </span>
      </div>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>{t("İşlem / boyut", "Operation / shape")}</th>
              <th>{t("Kütüphane", "Library")}</th>
              <th>{t("Medyan ms", "Median ms")}</th>
              <th>p95 ms</th>
              <th>{t("Maks. mutlak hata", "Max. absolute error")}</th>
            </tr>
          </thead>
          <tbody>
            {data.records.flatMap((row) =>
              row.results.map((r) => (
                <tr key={row.id + r.library}>
                  <td>
                    {row.operation}
                    <small className="cell-shape">{row.shape}</small>
                  </td>
                  <td>
                    <span
                      className={r.library === "NexusModel" ? "accent" : ""}
                    >
                      {r.library}
                    </span>
                  </td>
                  <td>{r.median_ms.toFixed(4)}</td>
                  <td>{r.p95_ms.toFixed(4)}</td>
                  <td>
                    {r.max_abs_error === 0
                      ? "0"
                      : r.max_abs_error.toExponential(2)}
                  </td>
                </tr>
              )),
            )}
          </tbody>
        </table>
      </div>
      <OptimizationResults lang={lang} />
      <div className="methodology-note">
        <Info size={20} />
        <div>
          <h3>
            {t(
              "Bu bir CPU işlem karşılaştırmasıdır.",
              "This is a CPU operation comparison.",
            )}
          </h3>
          <p>
            {t(
              "Forward-only GEMM (bias yok) ve ReLU; çıktı tamponları önceden ayrıldı. NumPy/PyTorch için Python API çağrı maliyeti, NexusModel için C++ çağrı maliyeti dahildir. Backward, autograd, bellek ayırma ve transfer hariçtir. p95, 5 çağrılık ortalamalardan oluşan 90 örneğin yüzdeliğidir. NumPy bir sayısal dizi kütüphanesidir; tam sinir ağı eğitim çerçevesi değildir.",
              "Forward-only GEMM (no bias) and ReLU with preallocated outputs. Python API dispatch is included for NumPy/PyTorch, C++ call overhead for NexusModel. Backward, autograd, allocation and transfers are excluded. p95 is the percentile of 90 samples, each averaging 5 calls. NumPy is a numerical array library, not a full neural-network training framework.",
            )}
          </p>
          <p>
            {t(
              "Tek makine, üç tur ve dönüşümlü kütüphane sırası; işletim sistemi zamanlaması ve sıcaklık sabitlenmemiştir. Her çıktı float64 referansına karşı kontrol edildi. Sürümler, BLAS backend bilgisi, ham süreler ve kaynak SHA-256 değerleri JSON dosyasında.",
              "One machine, three rounds with rotated framework order; OS scheduling and temperature were not controlled. Every output was checked against a float64 reference. Versions, BLAS backends, raw timings, and source SHA-256 hashes are in the JSON.",
            )}
          </p>
        </div>
      </div>
    </div>
  );
}
