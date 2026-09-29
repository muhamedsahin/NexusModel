"use client";
import Link from "next/link";
import { useEffect, useState } from "react";
import { Menu, X } from "lucide-react";
import { articles, groups, Locale, Section } from "@/lib/content";
export function DocsNav({ lang, slug }: { lang: Locale; slug: string }) {
  const [open, setOpen] = useState(false);
  return (
    <>
      <button
        className="docs-mobile-toggle"
        onClick={() => setOpen(!open)}
        aria-expanded={open}
      >
        {open ? <X size={16} /> : <Menu size={16} />}{" "}
        {lang === "tr" ? "Konu dizini" : "Topic index"}
      </button>
      <aside
        className={`docs-sidebar ${open ? "is-open" : ""}`}
        aria-label={
          lang === "tr" ? "Dokümantasyon konuları" : "Documentation topics"
        }
      >
        <Link className="docs-home" href={`/${lang}/docs/introduction/`}>
          NEXUSMODEL <span>DOCS</span>
        </Link>
        {groups.map((g, i) => (
          <div className="nav-group" key={i}>
            <h2>{g[lang]}</h2>
            {articles
              .filter((a) => a.group === i)
              .map((a) => (
                <Link
                  onClick={() => setOpen(false)}
                  aria-current={a.slug === slug ? "page" : undefined}
                  className={a.slug === slug ? "current" : ""}
                  key={a.slug}
                  href={`/${lang}/docs/${a.slug}/`}
                >
                  {a.title[lang]}
                </Link>
              ))}
          </div>
        ))}
        <div className="sidebar-bottom">
          <span>v1.0.0</span>
          <small>C++20 / FLOAT32</small>
        </div>
      </aside>
    </>
  );
}
export function Outline({
  sections,
  lang,
}: {
  sections: Section[];
  lang: Locale;
}) {
  const [active, setActive] = useState(sections[0]?.id);
  useEffect(() => {
    const io = new IntersectionObserver(
      (entries) => {
        for (const e of entries) if (e.isIntersecting) setActive(e.target.id);
      },
      { rootMargin: "-15% 0px -65% 0px" },
    );
    sections.forEach((s) => {
      const el = document.getElementById(s.id);
      if (el) io.observe(el);
    });
    return () => io.disconnect();
  }, [sections]);
  return (
    <aside className="article-outline">
      <span className="eyebrow">
        {lang === "tr" ? "BU SAYFADA" : "ON THIS PAGE"}
      </span>
      {sections.map((s) => (
        <a
          className={active === s.id ? "current" : ""}
          href={`#${s.id}`}
          key={s.id}
        >
          {s.title[lang]}
        </a>
      ))}
      <div className="outline-tip">
        ∂
        <p>
          {lang === "tr"
            ? "Her formülün arkasında okunabilir kaynak kod."
            : "Readable source code behind every formula."}
        </p>
      </div>
    </aside>
  );
}
