"use client";
import Link from "next/link";
import { useState } from "react";
import { Download, Info, Activity, BarChart3, FileText } from "lucide-react";
import { cpuRows, gpuRows } from "@/lib/benchmarks";
import { Locale } from "@/lib/types";
import comparisonData from "@/lib/comparison-results.json";
import { LibraryComparison } from "./library-comparison";
export function Benchmarks({ lang }: { lang: Locale }) {
  const [tab, setTab] = useState<"libraries" | "simd" | "gpu" | "cpu">(
    "libraries",
  );
  const [op, setOp] = useState("GEMM");
  const [selected, setSelected] = useState(1);
  const t = (tr: string, en: string) => (lang === "tr" ? tr : en);
  const choices = gpuRows.filter((r) => r.op === op);
  const row = choices[Math.min(selected, choices.length - 1)];
  const left = tab === "gpu" ? row.cpu : cpuRows[1].ms,
    right = tab === "gpu" ? row.gpu : cpuRows[2].ms;
  const ratio = left / right;
  function download() {
    if (tab === "libraries") {
      const a = document.createElement("a");
      a.href = "/benchmark-comparison.csv";
      a.download = "nexusmodel-libraries.csv";
      a.click();
      return;
    }
    const data =
      tab === "gpu"
        ? [
            "operation,shape,cpu_ms,gpu_ms,speedup",
            ...gpuRows.map(
              (r) =>
                `"${r.op}","${r.shape}",${r.cpu},${r.gpu},${(r.cpu / r.gpu).toFixed(2)}`,
            ),
          ]
        : [
            "operation,shape,ms,pass",
            ...cpuRows.map((r) => `"${r.op}","${r.shape}",${r.ms},"${r.pass}"`),
          ];
    const url = URL.createObjectURL(
      new Blob(["\ufeff" + data.join("\n")], {
        type: "text/csv;charset=utf-8",
      }),
    );
    const a = document.createElement("a");
    a.href = url;
    a.download = `nexusmodel-${tab}-published.csv`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  return (
    <main id="main" className="benchmark-page">
      <header className="benchmark-header">
        <span className="eyebrow">NEXUSMODEL / PERFORMANCE LAB</span>
        <h1>
          {t("Rakamlar konuşsun.", "Let the numbers talk.")}
          <span className="accent">_</span>
        </h1>
          <p>
          {t(
            "29 Eylül 2026 tabloları tek iş parçacığıdır. 21 Ekim 2026 akşamı bu Ryzen 7 5800H üzerinde GEMM avx2-4x16 kaldı: N=2048 Nexus 330, 362 ve 369 GFLOPS, TensorFlow 389. CPU Conv2d PyTorch 31.6 GFLOPS gerisindeydi; Nexus yaklaşık 4.9–6.8 GFLOPS. Depthwise AVX, PyTorch 4.93 ve TensorFlow 0.688 önündeydi. AVX-512 seçilmez. Dünya sıralaması yoktur.",
            "The 29 September 2026 tables are one thread. On the evening of 21 October 2026, on this Ryzen 7 5800H, GEMM kept avx2-4x16: N=2048 Nexus 330, 362, and 369 GFLOPS, TensorFlow 389. CPU Conv2d was behind PyTorch at 31.6 GFLOPS; Nexus was about 4.9 to 6.8 GFLOPS. Depthwise AVX was ahead of PyTorch 4.93 and TensorFlow 0.688. AVX-512 is not selected. There is no world ranking.",
          )}
        </p>
        <div className="bench-badges">
          {tab === "libraries" ? (
            <>
              <span>{comparisonData.metadata.cpu}</span>
              <span>CPU · FLOAT32</span>
              <span>NUMPY · PYTORCH · NEXUSMODEL</span>
            </>
          ) : (
            <>
              <span>MSVC 19.44 /O2</span>
              <span>RELEASE BUILD</span>
              {tab === "gpu" && (
                <>
                  <span>RTX 3070 LAPTOP</span>
                  <span>CUDA 13.4</span>
                </>
              )}
            </>
          )}
        </div>
      </header>
      <div className="bench-toolbar">
        <div
          className="segmented"
          role="tablist"
          aria-label={t("Karşılaştırma türü", "Comparison type")}
        >
          {(["libraries", "simd", "gpu", "cpu"] as const).map((v) => (
            <button
              role="tab"
              aria-selected={tab === v}
              aria-controls="bench-panel"
              id={`tab-${v}`}
              onClick={() => setTab(v)}
              onKeyDown={(e) => {
                if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
                  e.preventDefault();
                  const tabs = ["libraries", "simd", "gpu", "cpu"] as const;
                  const next =
                    tabs[
                      (tabs.indexOf(v) + (e.key === "ArrowRight" ? 1 : 3)) % 4
                    ];
                  setTab(next);
                  document.getElementById(`tab-${next}`)?.focus();
                }
              }}
              tabIndex={tab === v ? 0 : -1}
              key={v}
            >
              {v === "libraries"
                ? t("Kütüphaneler", "Libraries")
                : v === "simd"
                  ? "Scalar vs SIMD"
                  : v === "gpu"
                    ? t("CPU vs GPU · arşiv", "CPU vs GPU · archive")
                    : t("CPU katmanları", "CPU layers")}
            </button>
          ))}
        </div>
        <button className="button secondary compact" onClick={download}>
          <Download size={15} />
          CSV
        </button>
      </div>
      <div
        role="tabpanel"
        id="bench-panel"
        aria-labelledby={`tab-${tab}`}
        tabIndex={0}
      >
        {tab === "libraries" ? (
          <LibraryComparison lang={lang} />
        ) : (
          <>
            {tab !== "cpu" && (
              <div className="comparison-panel">
                <div className="comparison-main">
                  <div className="chart-top">
                    <span>
                      <BarChart3 size={16} />
                      {tab === "gpu"
                        ? `${row.op} / ${row.shape}`
                        : "LINEAR / 32 × 256 → 256"}
                    </span>
                    <span className="subtle-tag">
                      {t("DAHA DÜŞÜK DAHA İYİ", "LOWER IS BETTER")}
                    </span>
                  </div>
                  {tab === "gpu" && (
                    <div className="bench-selects">
                      <label>
                        {t("İşlem", "Operation")}
                        <select
                          aria-label={t("İşlem", "Operation")}
                          value={op}
                          onChange={(e) => {
                            setOp(e.target.value);
                            setSelected(0);
                          }}
                        >
                          {Array.from(new Set(gpuRows.map((r) => r.op))).map(
                            (x) => (
                              <option key={x}>{x}</option>
                            ),
                          )}
                        </select>
                      </label>
                      <label>
                        {t("Boyut", "Shape")}
                        <select
                          aria-label={t("Boyut", "Shape")}
                          value={Math.min(selected, choices.length - 1)}
                          onChange={(e) => setSelected(Number(e.target.value))}
                        >
                          {choices.map((r, i) => (
                            <option key={r.shape} value={i}>
                              {r.shape}
                            </option>
                          ))}
                        </select>
                      </label>
                    </div>
                  )}
                  <div className="bench-bar-group">
                    <div className="bar-label">
                      <strong>{tab === "gpu" ? "CPU" : "Scalar"}</strong>
                      <span>
                        {left.toLocaleString(lang, {
                          maximumFractionDigits: 4,
                        })}{" "}
                        ms
                      </span>
                    </div>
                    <div className="bar-track large">
                      <i style={{ width: "100%" }} />
                    </div>
                    <div className="bar-label accent">
                      <strong>
                        {tab === "gpu" ? "GPU" : "NexusModel · SIMD"}
                      </strong>
                      <span>
                        {right.toLocaleString(lang, {
                          maximumFractionDigits: 4,
                        })}{" "}
                        ms
                      </span>
                    </div>
                    <div className="bar-track large">
                      <i
                        className="accent-bar"
                        style={{ width: `${(right / left) * 100}%` }}
                      />
                    </div>
                    <div className="axis">
                      <span>0 ms</span>
                      <span>
                        {(left / 2).toLocaleString(lang, {
                          maximumFractionDigits: 2,
                        })}
                      </span>
                      <span>{left} ms</span>
                    </div>
                  </div>
                  <p className="chart-note">
                    {tab === "gpu"
                      ? t(
                          "Cihazda hazır veri · yalnız forward çekirdeği · transferler hariç. GEMM, cuBLAS ile ölçüldü.",
                          "Resident device data · forward kernel only · transfers excluded. GEMM measured with cuBLAS.",
                        )
                      : t(
                          "Isınma sonrası forward + backward ortalaması. Aynı katman ve aynı tensör boyutları.",
                          "Mean forward + backward after warmup. Same layer and tensor dimensions.",
                        )}
                  </p>
                </div>
                <div className="comparison-result">
                  <Activity size={28} />
                  <span className="eyebrow">
                    {t("ÖLÇÜLEN ORAN", "MEASURED RATIO")}
                  </span>
                  <strong key={`${tab}-${op}-${selected}`}>
                    {ratio.toFixed(2)}
                    <span>×</span>
                  </strong>
                  <p>
                    {t(
                      "Referans süre / hızlandırılmış süre",
                      "Reference time / accelerated time",
                    )}
                  </p>
                  <small>
                    {t(
                      "Bu iş yüküne özeldir. Uçtan uca eğitim hızlanması değildir.",
                      "Specific to this workload. Not end-to-end training acceleration.",
                    )}
                  </small>
                </div>
              </div>
            )}
            <div className="results-heading">
              <h2>
                {tab === "gpu"
                  ? t("CPU / GPU sonuçları", "CPU / GPU results")
                  : t("CPU katman sonuçları", "CPU layer results")}
              </h2>
              <span>
                {t("YAYIMLANMIŞ VERİ", "PUBLISHED DATA")} ·{" "}
                {tab === "gpu" ? gpuRows.length : cpuRows.length}{" "}
                {t("ölçüm", "measurements")}
              </span>
            </div>
            <div className="table-wrap benchmark-table">
              <table>
                <thead>
                  <tr>
                    <th>{t("İşlem", "Operation")}</th>
                    <th>{t("Boyut", "Shape")}</th>
                    {tab === "gpu" ? (
                      <>
                        <th>CPU ms</th>
                        <th>GPU ms</th>
                        <th>{t("Oran", "Ratio")}</th>
                      </>
                    ) : (
                      <>
                        <th>{t("Kapsam", "Scope")}</th>
                        <th>ms</th>
                      </>
                    )}
                  </tr>
                </thead>
                <tbody>
                  {tab === "gpu"
                    ? gpuRows.map((r, i) => (
                        <tr key={i}>
                          <td>{r.op}</td>
                          <td>{r.shape}</td>
                          <td>{r.cpu}</td>
                          <td className="accent">{r.gpu}</td>
                          <td>{(r.cpu / r.gpu).toFixed(2)}×</td>
                        </tr>
                      ))
                    : cpuRows.map((r, i) => (
                        <tr key={i}>
                          <td>{r.op}</td>
                          <td>{r.shape}</td>
                          <td>{r.pass}</td>
                          <td className={r.op.includes("SIMD") ? "accent" : ""}>
                            {r.ms}
                          </td>
                        </tr>
                      ))}
                </tbody>
              </table>
            </div>
          </>
        )}
      </div>
      {tab !== "libraries" && (
        <div className="methodology-note">
          <Info size={20} />
          <div>
            <h3>
              {t(
                "Ölçüm koşulları sonuçların bir parçasıdır.",
                "Measurement conditions are part of the result.",
              )}
            </h3>
            <p>
              {t(
                "Kaynak: benchmarks/RESULTS.md. CPU katman milisaniyeleri 29 Eylül 2026 ölçümüdür. Conv2D satırı, 21 Ekim akşamının GFLOPS sütunu değildir; o akşam CPU Conv2d PyTorch 31.6 GFLOPS gerisindeydi ve Nexus yaklaşık 4.9–6.8 GFLOPS idi. GPU tablosu eski arşivdir. GPU süreleri cudaEvent ile ölçülür, H2D/D2H aktarımı ve ayırma hariçtir. Bu sayfa canlı benchmark çalıştırmaz.",
                "Source: benchmarks/RESULTS.md. CPU layer milliseconds are the 29 September 2026 measurement. The Conv2D row is not the 21 October GFLOPS column; that evening CPU Conv2d was behind PyTorch at 31.6 GFLOPS and Nexus was about 4.9 to 6.8 GFLOPS. The GPU table is a historical archive. GPU timings use cudaEvent, excluding H2D/D2H transfers and allocation. This page does not run live benchmarks.",
              )}
            </p>
            <div className="actions">
              <Link href={`/${lang}/docs/benchmark-methodology/`}>
                {t("Metodolojiyi oku", "Read the methodology")} ↗
              </Link>
              <a href="/benchmark-results.md" download>
                <FileText size={14} />
                {t("Kaynak raporu indir", "Download source report")}
              </a>
            </div>
          </div>
        </div>
      )}
    </main>
  );
}
