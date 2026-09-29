"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import {
  Search,
  Sun,
  Moon,
  Menu,
  X,
  Command,
  BookOpen,
  ChevronRight,
  Pause,
  Play,
} from "lucide-react";
import { articles, Locale } from "@/lib/content";
export function Brand() {
  return (
    <span className="brand">
      <svg viewBox="0 0 32 32" width="29" height="29" aria-hidden="true">
        <path d="M4 26V6h6l12 15V6h6v20h-6L10 11v15Z" fill="currentColor" />
        <path d="M4 26 28 6" stroke="var(--bg)" strokeWidth="2" />
      </svg>
      Nexus<span>Model</span>
    </span>
  );
}
export function Shell({
  lang,
  children,
}: {
  lang: Locale;
  children: React.ReactNode;
}) {
  const path = usePathname();
  const [theme, setTheme] = useState("dark");
  const [menu, setMenu] = useState(false);
  const [search, setSearch] = useState(false);
  const [q, setQ] = useState("");
  const [paused, setPaused] = useState(false);
  const [loading, setLoading] = useState(false);
  const dialog = useRef<HTMLDialogElement>(null);
  const trigger = useRef<HTMLButtonElement>(null);
  const t = (tr: string, en: string) => (lang === "tr" ? tr : en);
  useEffect(() => {
    document.documentElement.lang = lang;
    try {
      localStorage.setItem("nx-locale", lang);
    } catch {}
  }, [lang]);
  useEffect(() => {
    setTheme(document.documentElement.dataset.theme || "dark");
    const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
    setPaused(reduced);
    document.documentElement.dataset.paused = String(reduced);
    try {
      if (!sessionStorage.getItem("nx-intro") && !reduced) {
        setLoading(true);
        sessionStorage.setItem("nx-intro", "1");
      }
    } catch {}
    const timer = setTimeout(() => setLoading(false), 1250);
    return () => clearTimeout(timer);
  }, []);
  useEffect(() => {
    const fn = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "k") {
        e.preventDefault();
        setSearch((v) => !v);
      }
    };
    window.addEventListener("keydown", fn);
    return () => window.removeEventListener("keydown", fn);
  }, []);
  useEffect(() => {
    if (search) dialog.current?.showModal();
    else if (dialog.current?.open) dialog.current.close();
  }, [search]);
  useEffect(() => {
    setMenu(false);
    setSearch(false);
    setQ("");
  }, [path]);
  function toggleTheme() {
    const v = theme === "dark" ? "light" : "dark";
    setTheme(v);
    document.documentElement.dataset.theme = v;
    try {
      localStorage.setItem("nx-theme", v);
    } catch {}
    window.dispatchEvent(new Event("nx-theme"));
  }
  function toggleMotion() {
    const v = !paused;
    setPaused(v);
    document.documentElement.dataset.paused = String(v);
    window.dispatchEvent(new Event("nx-motion"));
  }
  const found = articles
    .filter((a) =>
      `${a.title[lang]} ${a.description[lang]} ${a.sections.map((s) => s.body[lang]).join(" ")}`
        .toLocaleLowerCase(lang)
        .includes(q.toLocaleLowerCase(lang)),
    )
    .slice(0, 9);
  return (
    <>
      <a className="skip-link" href="#main">
        {t("İçeriğe geç", "Skip to content")}
      </a>
      {loading && (
        <div className="intro" aria-hidden="true">
          <div className="intro-symbol">
            <i />
            <i />
            <i />
          </div>
          <Brand />
          <span>
            C O M P U T A T I O N , &nbsp; M A D E &nbsp; V I S I B L E
          </span>
          <div className="intro-line" />
        </div>
      )}
      <header className="site-header">
        <div className="header-inner">
          <Link href={`/${lang}/`} aria-label="NexusModel">
            <Brand />
          </Link>
          <span className="version">v1.0.0</span>
          <nav className="desktop-nav">
            <Link
              className={path.includes("/docs") ? "active" : ""}
              href={`/${lang}/docs/introduction/`}
            >
              {t("Dokümantasyon", "Documentation")}
            </Link>
            <Link
              className={path.includes("/benchmarks") ? "active" : ""}
              href={`/${lang}/benchmarks/`}
            >
              Benchmark
            </Link>
            <Link href={`/${lang}/#architecture`}>
              {t("Mimari", "Architecture")}
            </Link>
          </nav>
          <div className="header-tools">
            <button
              ref={trigger}
              className="search-trigger"
              onClick={() => setSearch(true)}
              aria-label={t("Dokümanlarda ara", "Search documentation")}
            >
              <Search size={15} />
              <span>{t("Ara", "Search")}</span>
              <kbd>⌘ K</kbd>
            </button>
            <Link
              className="lang-switch"
              href={path.replace(/^\/(tr|en)/, lang === "tr" ? "/en" : "/tr")}
              aria-label={t("Switch to English", "Türkçeye geç")}
            >
              <span className={lang === "tr" ? "selected" : ""}>TR</span>
              <b>/</b>
              <span className={lang === "en" ? "selected" : ""}>EN</span>
            </Link>
            <button
              className="icon-button"
              onClick={toggleTheme}
              aria-label={t("Temayı değiştir", "Toggle theme")}
            >
              {theme === "dark" ? <Sun size={18} /> : <Moon size={18} />}
            </button>
            <button
              className="icon-button motion-button"
              onClick={toggleMotion}
              aria-pressed={paused}
              aria-label={t(
                paused ? "Animasyonları oynat" : "Animasyonları durdur",
                paused ? "Play animations" : "Pause animations",
              )}
            >
              {paused ? <Play size={15} /> : <Pause size={15} />}
            </button>
            <button
              className="icon-button mobile-menu"
              onClick={() => setMenu(!menu)}
              aria-expanded={menu}
              aria-label={t("Menüyü aç", "Open menu")}
            >
              {menu ? <X size={20} /> : <Menu size={20} />}
            </button>
          </div>
        </div>
        {menu && (
          <nav className="mobile-nav">
            <Link href={`/${lang}/docs/introduction/`}>
              {t("Dokümantasyon", "Documentation")}
            </Link>
            <Link href={`/${lang}/benchmarks/`}>Benchmark</Link>
            <Link href={`/${lang}/#architecture`}>
              {t("Mimari", "Architecture")}
            </Link>
            <button className="mobile-motion" onClick={toggleMotion}>
              {paused ? <Play size={15} /> : <Pause size={15} />}{" "}
              {t(
                paused ? "Animasyonları oynat" : "Animasyonları durdur",
                paused ? "Play animations" : "Pause animations",
              )}
            </button>
          </nav>
        )}
      </header>
      {children}
      <footer className="site-footer">
        <div>
          <Link href={`/${lang}/`}>
            <Brand />
          </Link>
          <p>
            {t(
              "Hesaplamanın her adımı, sizin kontrolünüzde.",
              "Every step of computation, under your control.",
            )}
          </p>
        </div>
        <div>
          <Link href={`/${lang}/docs/quickstart/`}>
            {t("Başlangıç", "Get started")}
          </Link>
          <Link href={`/${lang}/docs/testing/`}>
            {t("Sınırlar ve doğrulama", "Limits & validation")}
          </Link>
          <Link href={`/${lang}/benchmarks/`}>Benchmark</Link>
        </div>
        <span className="footer-note">
          C++20 · FLOAT32 · EXPLICIT BACKWARD
          <br />
          NEXUS ECOSYSTEM / 01
        </span>
      </footer>
      <dialog
        ref={dialog}
        className="search-dialog"
        onCancel={() => setSearch(false)}
        onClose={() => {
          setSearch(false);
          trigger.current?.focus();
        }}
        onClick={(e) => {
          if (e.target === e.currentTarget) setSearch(false);
        }}
      >
        <div className="search-input">
          <Search size={20} />
          <input
            autoFocus
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder={t(
              "Bir katman, kavram veya API ara…",
              "Search a layer, concept, or API…",
            )}
            aria-label={t("Arama", "Search")}
          />
          <button
            className="icon-button"
            onClick={() => setSearch(false)}
            aria-label={t("Kapat", "Close")}
          >
            <X size={18} />
          </button>
        </div>
        <div className="search-results">
          <p className="eyebrow">
            {t("DOKÜMANTASYON", "DOCUMENTATION")} · {found.length}
          </p>
          {found.map((a) => (
            <Link
              key={a.slug}
              href={`/${lang}/docs/${a.slug}/`}
              onClick={() => setSearch(false)}
            >
              <BookOpen size={18} />
              <span>
                <strong>{a.title[lang]}</strong>
                <small>{a.description[lang]}</small>
              </span>
              <ChevronRight size={16} />
            </Link>
          ))}
          {!found.length && (
            <p className="empty-result">
              {t(
                "Sonuç bulunamadı. “Tensor” veya “Linear” deneyin.",
                "No results. Try “Tensor” or “Linear”.",
              )}
            </p>
          )}
        </div>
        <div className="search-hint">
          <Command size={13} /> K <span>{t("arama", "search")}</span>
          <kbd>ESC</kbd>
          {t("kapat", "close")}
        </div>
      </dialog>
    </>
  );
}
