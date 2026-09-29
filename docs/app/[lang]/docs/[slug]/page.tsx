import Link from "next/link";
import { notFound } from "next/navigation";
import {
  BookOpen,
  Code2,
  ChevronRight,
  ChevronLeft,
  FileText,
} from "lucide-react";
import katex from "katex";
import { articles, getArticle, groups, Locale } from "@/lib/content";
import { Code } from "@/components/ui";
import { DocsNav, Outline } from "@/components/docs-nav";
export function generateStaticParams() {
  return articles.map((a) => ({ slug: a.slug }));
}
export const dynamicParams = false;
export async function generateMetadata({
  params,
}: {
  params: Promise<{ lang: Locale; slug: string }>;
}) {
  const { lang, slug } = await params;
  const a = getArticle(slug);
  return { title: a?.title[lang], description: a?.description[lang] };
}
export default async function Doc({
  params,
}: {
  params: Promise<{ lang: Locale; slug: string }>;
}) {
  const { lang, slug } = await params;
  const a = getArticle(slug);
  if (!a) notFound();
  const idx = articles.indexOf(a);
  const previous = articles[idx - 1],
    next = articles[idx + 1];
  const t = (tr: string, en: string) => (lang === "tr" ? tr : en);
  const words = a.sections.reduce(
    (n, s) => n + s.body[lang].split(" ").length,
    0,
  );
  return (
    <main id="main" className="docs-layout">
      <DocsNav lang={lang} slug={slug} />
      <article className="doc-article">
        <div className="breadcrumbs">
          <BookOpen size={14} />
          <Link href={`/${lang}/docs/introduction/`}>Docs</Link>
          <ChevronRight size={13} />
          <span>{groups[a.group][lang]}</span>
        </div>
        <header className="article-header">
          <span className="eyebrow">
            NEXUSMODEL / {String(idx + 1).padStart(2, "0")}
          </span>
          <h1>{a.title[lang]}</h1>
          <p>{a.description[lang]}</p>
          <div className="article-meta">
            <span>
              <FileText size={13} />
              {Math.max(2, Math.ceil(words / 120))} {t("dk okuma", "min read")}
            </span>
            <span>v1.0.0</span>
            <span>C++20</span>
          </div>
        </header>
        <div className="source-callout">
          <Code2 size={19} />
          <div>
            <strong>
              {t("Doğrudan uygulamadan", "From the implementation")}
            </strong>
            <a
              href={`/source/${a.source}.txt`}
              target="_blank"
              rel="noreferrer"
            >
              {a.source}
            </a>
          </div>
        </div>
        {a.sections.map((s, i) => (
          <section className="article-section" id={s.id} key={s.id}>
            <h2>
              <a href={`#${s.id}`}>
                <span className="section-number">
                  {String(i + 1).padStart(2, "0")}
                </span>
                {s.title[lang]}
                <span className="hash">#</span>
              </a>
            </h2>
            <p>{s.body[lang]}</p>
            {s.formula && (
              <div
                className="math-block"
                dangerouslySetInnerHTML={{
                  __html: katex.renderToString(s.formula, {
                    displayMode: true,
                    throwOnError: true,
                    output: "htmlAndMathml",
                  }),
                }}
              />
            )}
            {s.code && (
              <Code
                code={s.code}
                lang={lang}
                label={
                  slug === "installation" || slug === "benchmark-methodology"
                    ? "TERMINAL"
                    : "C++20"
                }
              />
            )}
            {s.table && (
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      {s.table.headers.map((h, j) => (
                        <th key={j}>{h[lang]}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {s.table.rows.map((row, j) => (
                      <tr key={j}>
                        {row.map((v, k) => (
                          <td key={k}>{v}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </section>
        ))}
        {slug === "library-comparison" && (
          <div className="official-sources">
            <span className="eyebrow">
              {t("RESMÎ API KAYNAKLARI", "OFFICIAL API REFERENCES")}
            </span>
            <a
              href="https://numpy.org/doc/2.4/reference/generated/numpy.matmul.html"
              target="_blank"
              rel="noreferrer"
            >
              NumPy · matmul
            </a>
            <a
              href="https://docs.pytorch.org/docs/2.14/generated/torch.mm.html"
              target="_blank"
              rel="noreferrer"
            >
              PyTorch · torch.mm
            </a>
            <a
              href="https://docs.pytorch.org/docs/stable/threading_environment_variables.html"
              target="_blank"
              rel="noreferrer"
            >
              PyTorch · threading
            </a>
          </div>
        )}
        <div className="article-end">
          <span>◆</span>
          {t(
            "Kaynak kodla birlikte okuyun. API davranışı mevcut uygulamayı esas alır.",
            "Read alongside the source. API behavior reflects the current implementation.",
          )}
        </div>
        <nav
          className="doc-pagination"
          aria-label={t("Önceki ve sonraki konu", "Previous and next topics")}
        >
          {previous ? (
            <Link href={`/${lang}/docs/${previous.slug}/`}>
              <small>
                <ChevronLeft size={13} />
                {t("ÖNCEKİ", "PREVIOUS")}
              </small>
              <strong>{previous.title[lang]}</strong>
            </Link>
          ) : (
            <div />
          )}
          {next && (
            <Link href={`/${lang}/docs/${next.slug}/`}>
              <small>
                {t("SONRAKİ", "NEXT")}
                <ChevronRight size={13} />
              </small>
              <strong>{next.title[lang]}</strong>
            </Link>
          )}
        </nav>
      </article>
      <Outline sections={a.sections} lang={lang} />
    </main>
  );
}
