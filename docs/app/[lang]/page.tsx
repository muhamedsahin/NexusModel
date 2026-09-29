import Link from "next/link";
import { cpuRows } from "@/lib/benchmarks";
import {
  Cpu,
  Layers,
  Workflow,
  Braces,
  Box,
  Network,
  ScanLine,
  BookOpen,
  Activity,
  Check,
} from "lucide-react";
import { TensorScene } from "@/components/tensor-scene";
import { Code, Reveal } from "@/components/ui";
import { Locale, quickCode } from "@/lib/content";
export default async function Home({
  params,
}: {
  params: Promise<{ lang: Locale }>;
}) {
  const { lang } = await params;
  const t = (tr: string, en: string) => (lang === "tr" ? tr : en);
  const families = [
    {
      name: "Linear",
      icon: Braces,
      slug: "linear",
      desc: t(
        "Son eksenden başlayın. İleri ve geri yönde tam kontrol.",
        "Start with the final axis. Full control in both directions.",
      ),
      formula: "Y = XWᵀ + b",
      tags: "MLP / PROJECTION",
    },
    {
      name: t("Evrişim", "Convolution"),
      icon: ScanLine,
      slug: "convolution",
      desc: t(
        "1D, 2D, 3D. Gruplu ve depthwise evrişimler.",
        "1D, 2D, 3D. Grouped and depthwise convolutions.",
      ),
      formula: "im2col → GEMM",
      tags: "CNN / VISION",
    },
    {
      name: "Attention",
      icon: Network,
      slug: "attention",
      desc: t(
        "Çoklu başlar. Yerel MicroTape. Açık gradyan akışı.",
        "Multiple heads. Local MicroTape. Explicit gradient flow.",
      ),
      formula: "softmax(QKᵀ / √d)V",
      tags: "TRANSFORMER / SEQUENCE",
    },
    {
      name: "RNN · LSTM · GRU",
      icon: Workflow,
      slug: "recurrent",
      desc: t(
        "Zaman boyunca analitik türev. Kesilebilir BPTT.",
        "Analytic derivatives through time. Truncated BPTT.",
      ),
      formula: "hₜ = f(xₜ, hₜ₋₁)",
      tags: "RECURRENT / TEMPORAL",
    },
  ];
  return (
    <main id="main">
      <section className="hero">
        <div className="hero-grid" />
        <div className="hero-copy">
          <div className="release-pill">
            <span /> C++20 NEURAL NETWORK LIBRARY <b>1.0</b>
          </div>
          <h1>
            {t("Zekânın", "Intelligence.")}
            <br />
            {t("temelinde.", "At its core.")}
            <span className="accent-period">_</span>
          </h1>
          <p className="hero-lead">
            {t(
              "Modelinizi kurun. Hesabı görün.",
              "Build your model. See the computation.",
            )}
            <br />
            <span>
              {t(
                "Her gradyanın kontrolünü elinize alın.",
                "Take control of every gradient.",
              )}
            </span>
          </p>
          <p className="hero-description">
            {t(
              "Yüksek performanslı sinir ağı katmanları, açık geri yayılım ve donanıma yakın C++ gücü. Sade bir arayüzün arkasında, tüm matematik sizin.",
              "High-performance neural network layers, explicit backward, and C++ close to the hardware. A simple interface. All the mathematics, yours.",
            )}
          </p>
          <div className="actions">
            <Link className="button primary" href={`/${lang}/docs/quickstart/`}>
              <BookOpen size={17} />
              {t("Oluşturmaya başlayın", "Start building")}
            </Link>
            <Link className="button secondary" href={`/${lang}/benchmarks/`}>
              {t("Performansı keşfet", "Explore performance")}
              <Activity size={16} />
            </Link>
          </div>
          <div className="hero-footnote">
            <span>HOST FLOAT32</span>
            <i />
            <span>AVX2 / AVX-512</span>
            <i />
            <span>EXPLICIT BACKWARD</span>
          </div>
        </div>
        <TensorScene lang={lang} />
        <div className="hero-bottom">
          <span>NEXUS ECOSYSTEM / MODEL LAYER</span>
          <a href="#architecture">
            {t("KEŞFETMEK İÇİN KAYDIR", "SCROLL TO EXPLORE")}{" "}
            <span className="scroll-mark" />
          </a>
          <span>DESIGNED FOR C++20</span>
        </div>
      </section>
      <div className="metrics-strip">
        <div>
          <span className="metric-value">
            {(cpuRows[1].ms / cpuRows[2].ms).toFixed(2)}<span>×</span>
          </span>
          <p>{t("SIMD / skalar hız oranı", "SIMD / scalar speed ratio")}</p>
          <small>Linear · B32 · 256 → 256</small>
        </div>
        <div>
          <span className="metric-value">
            {cpuRows[2].ms.toFixed(3)}<span>ms</span>
          </span>
          <p>{t("İleri + geri geçiş", "Forward + backward")}</p>
          <small>{t("Aynı Linear iş yükü", "Same Linear workload")}</small>
        </div>
        <div>
          <span className="metric-value">
            64<span>byte</span>
          </span>
          <p>{t("Hizalı tensör belleği", "Aligned tensor storage")}</p>
          <small>
            {t("Float32 · paylaşımlı depolama", "Float32 · shared storage")}
          </small>
        </div>
        <div>
          <span className="metric-value">
            01<span>API</span>
          </span>
          <p>
            {t(
              "Tüm katmanlar için bir sözleşme",
              "One contract for every layer",
            )}
          </p>
          <small>forward(x) / backward(dy)</small>
        </div>
      </div>
      <section id="architecture" className="section architecture">
        <Reveal>
          <div className="section-heading">
            <div>
              <span className="eyebrow">
                01 / {t("MİMARİ", "ARCHITECTURE")}
              </span>
              <h2>
                {t("Kara kutuyu açın.", "Open the black box.")}
                <br />
                <span>{t("Her adımı anlayın.", "Understand every step.")}</span>
              </h2>
            </div>
            <p>
              {t(
                "Veri, katman, kayıp ve güncelleme. Sorumlulukları ayrılmış bir ekosistem, açıkça görünen bir hesaplama akışı.",
                "Data, layers, loss, and updates. An ecosystem with clear responsibilities and a computation flow you can follow.",
              )}
            </p>
          </div>
        </Reveal>
        <Reveal>
          <div className="flow-diagram">
            {["NexusData", "NexusModel", "NexusLoss", "NexusOptim"].map(
              (name, i) => (
                <div
                  className={i === 1 ? "flow-node focused" : "flow-node"}
                  key={name}
                >
                  <span className="node-index">0{i + 1}</span>
                  {i === 1 ? (
                    <Box size={29} />
                  ) : i === 0 ? (
                    <Layers size={25} />
                  ) : i === 2 ? (
                    <Activity size={25} />
                  ) : (
                    <Workflow size={25} />
                  )}
                  <strong>{name}</strong>
                  <code>
                    {
                      [
                        "x: Tensor",
                        "y = model.forward(x)",
                        "dL / dy",
                        "θ ← optimizer.step()",
                      ][i]
                    }
                  </code>
                  <small>
                    {
                      [
                        t("Veriyi hazırlayın", "Prepare data"),
                        t("Hesabı çalıştırın", "Run computation"),
                        t("Gradyanı üretin", "Produce gradients"),
                        t("Ağırlıkları güncelleyin", "Update weights"),
                      ][i]
                    }
                  </small>
                </div>
              ),
            )}
          </div>
          <div className="backward-path">
            <span>∂L / ∂θ</span>
            <i />
            <p>
              {t(
                "Açık geri yayılım. Paylaşılan global teyp gerekmez.",
                "Explicit backward. No shared global tape required.",
              )}
            </p>
          </div>
        </Reveal>
        <div className="principles">
          <Reveal>
            <Cpu />
            <h3>{t("Donanıma yakın.", "Close to the hardware.")}</h3>
            <p>
              {t(
                "Çalışma zamanında seçilen SIMD çekirdekleri. 64 bayt hizalı, tekrar kullanılabilen tamponlar.",
                "Runtime-selected SIMD kernels. 64-byte aligned, reusable buffers.",
              )}
            </p>
            <Link href={`/${lang}/docs/simd/`}>
              {t("CPU mimarisini incele", "Explore CPU architecture")}{" "}
              <span>↗</span>
            </Link>
          </Reveal>
          <Reveal>
            <Workflow />
            <h3>{t("Türevler görünür.", "Derivatives, visible.")}</h3>
            <p>
              {t(
                "Katmanlarda analitik backward. Attention içinde küçük, katmana özel MicroTape.",
                "Analytic backward in layers. A small, layer-local MicroTape inside attention.",
              )}
            </p>
            <Link href={`/${lang}/docs/autograd/`}>
              {t("Geri yayılımı anla", "Understand backward")} <span>↗</span>
            </Link>
          </Reveal>
          <Reveal>
            <Braces />
            <h3>{t("Birleşmeye hazır.", "Made to compose.")}</h3>
            <p>
              {t(
                "Sequential ile birleştirin. Module ile genişletin. Matematikten uygulamaya doğrudan geçin.",
                "Compose with Sequential. Extend with Module. Go directly from mathematics to implementation.",
              )}
            </p>
            <Link href={`/${lang}/docs/containers/`}>
              {t("Modülleri keşfet", "Explore modules")} <span>↗</span>
            </Link>
          </Reveal>
        </div>
      </section>
      <section className="section layer-section">
        <Reveal>
          <div className="section-heading">
            <div>
              <span className="eyebrow">
                02 / {t("YAPI TAŞLARI", "BUILDING BLOCKS")}
              </span>
              <h2>
                {t("Fikirden mimariye.", "From idea to architecture.")}
                <br />
                <span>{t("Katman katman.", "Layer by layer.")}</span>
              </h2>
            </div>
            <Link className="text-link" href={`/${lang}/docs/linear/`}>
              {t("Katman referansını aç", "Open the layer reference")}{" "}
              <BookOpen size={16} />
            </Link>
          </div>
        </Reveal>
        <div className="layer-cards">
          {families.map((f, i) => (
            <Reveal key={f.name}>
              <Link href={`/${lang}/docs/${f.slug}/`} className="layer-card">
                <div className="card-top">
                  <f.icon size={23} />
                  <span>0{i + 1}</span>
                </div>
                <div className="layer-formula">{f.formula}</div>
                <span className="eyebrow">{f.tags}</span>
                <h3>{f.name}</h3>
                <p>{f.desc}</p>
                <span className="card-corner">↗</span>
              </Link>
            </Reveal>
          ))}
        </div>
        <p className="also-included">
          {t("Ayrıca:", "Also included:")}{" "}
          <Link href={`/${lang}/docs/normalization/`}>Normalization</Link>
          <span>·</span>
          <Link href={`/${lang}/docs/activations/`}>11 activations</Link>
          <span>·</span>
          <Link href={`/${lang}/docs/pooling/`}>Pooling</Link>
          <span>·</span>
          <Link href={`/${lang}/docs/embedding/`}>Embedding</Link>
          <span>·</span>
          <Link href={`/${lang}/docs/dropout/`}>Dropout</Link>
        </p>
      </section>
      <section className="section home-performance">
        <Reveal className="performance-copy">
          <span className="eyebrow">
            03 / {t("ÖLÇÜLMÜŞ PERFORMANS", "MEASURED PERFORMANCE")}
          </span>
          <h2>
            {t("Daha az bekleme.", "Less waiting.")}
            <br />
            <span>{t("Daha çok hesap.", "More computing.")}</span>
          </h2>
          <p>
            {t(
              "Aynı iş yükü. Aynı boyutlar. SIMD ile skalar yolu yan yana görün. Rakamların arkasındaki koşullar her zaman açık.",
              "Same workload. Same dimensions. Compare SIMD and scalar paths side by side. The conditions behind every number are always visible.",
            )}
          </p>
          <Link className="button secondary" href={`/${lang}/benchmarks/`}>
            {t("NexusModel · NumPy · PyTorch", "NexusModel · NumPy · PyTorch")}
            <Activity size={16} />
          </Link>
        </Reveal>
        <Reveal className="home-chart">
          <div className="chart-top">
            <span>LINEAR · 32 × 256 → 256</span>
            <span className="subtle-tag">CPU</span>
          </div>
          <div className="chart-stat">
            {(cpuRows[1].ms / cpuRows[2].ms).toFixed(2)}<span>×</span>
            <small>{t("ölçülen hız oranı", "measured speed ratio")}</small>
          </div>
          <div className="bar-label">
            <span>{t("Skalar", "Scalar")}</span>
            <strong>{cpuRows[1].ms.toFixed(3)} ms</strong>
          </div>
          <div className="bar-track">
            <i style={{ width: "100%" }} />
          </div>
          <div className="bar-label accent">
            <span>NexusModel · SIMD</span>
            <strong>{cpuRows[2].ms.toFixed(3)} ms</strong>
          </div>
          <div className="bar-track">
            <i className="accent-bar" style={{ width: `${cpuRows[2].ms / cpuRows[1].ms * 100}%` }} />
          </div>
          <div className="chart-note">
            {t(
              "Forward + backward · Release /O2 · MSVC 19.44. Kaynak: depodaki benchmark raporu.",
              "Forward + backward · Release /O2 · MSVC 19.44. Source: repository benchmark report.",
            )}
          </div>
        </Reveal>
      </section>
      <section className="section build-section">
        <Reveal>
          <div className="section-heading">
            <div>
              <span className="eyebrow">04 / HELLO, NEXUSMODEL</span>
              <h2>
                {t("Az kod.", "Less code.")}
                <br />
                <span>{t("Tam kontrol.", "Full control.")}</span>
              </h2>
            </div>
            <p>
              {t(
                "Modeli tanımlayın, girdiyi geçirin, dış gradyanı geri gönderin. İlk çalışan örneğiniz, tam burada.",
                "Define the model, pass the input, send the external gradient back. Your first working example, right here.",
              )}
            </p>
          </div>
        </Reveal>
        <div className="build-grid">
          <Reveal>
            <Code code={quickCode} label="main.cpp" lang={lang} />
          </Reveal>
          <Reveal className="build-notes">
            <span className="eyebrow">THE DEVELOPER EXPERIENCE</span>
            <h3>
              {t(
                "Matematikle aynı dili konuşan bir API.",
                "An API that speaks the language of mathematics.",
              )}
            </h3>
            {[
              t(
                "C++20 ve standart CMake derlemesi",
                "C++20 and a standard CMake build",
              ),
              t(
                "Açık girdi ve çıktı şekilleri",
                "Explicit input and output shapes",
              ),
              t(
                "Doğrudan erişilebilir parametre gradyanları",
                "Directly accessible parameter gradients",
              ),
              t(
                "Kaynak kodla eşleşen ayrıntılı rehberler",
                "Detailed guides grounded in source code",
              ),
            ].map((x) => (
              <p className="check-line" key={x}>
                <Check size={16} />
                {x}
              </p>
            ))}
            <Link
              href={`/${lang}/docs/installation/`}
              className="button primary"
            >
              {t("Kurulum rehberini aç", "Open the installation guide")}
            </Link>
            <p className="small-note">
              {t(
                "Örnek L = sum(y) kullanır. Gerçek eğitim için kayıp ve optimizer ekleyin.",
                "Example uses L = sum(y). Add a loss and optimizer for real training.",
              )}
            </p>
          </Reveal>
        </div>
      </section>
      <section className="closing section">
        <Reveal>
          <span className="eyebrow">BUILT TO BE UNDERSTOOD</span>
          <h2>
            {t("Sadece kullanmayın.", "Don’t just use it.")}
            <br />
            <em>{t("İçini öğrenin.", "Understand it.")}</em>
          </h2>
          <p>
            {t(
              "Tensör belleğinden attention matematiğine. Tek bir satırdan, bütün mimariye.",
              "From tensor memory to attention mathematics. From a single line to the whole architecture.",
            )}
          </p>
          <Link className="button primary" href={`/${lang}/docs/introduction/`}>
            <BookOpen size={17} />
            {t("Dokümantasyonu keşfet", "Explore the documentation")}
          </Link>
        </Reveal>
      </section>
    </main>
  );
}

