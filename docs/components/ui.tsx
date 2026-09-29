"use client";
import { useEffect, useRef, useState } from "react";
import { Check, Copy } from "lucide-react";
import type { Locale } from "@/lib/types";
export function Code({
  code,
  label = "C++20",
  lang = "en",
}: {
  code: string;
  label?: string;
  lang?: Locale;
}) {
  const [state, setState] = useState("");
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(
    () => () => {
      if (timer.current) clearTimeout(timer.current);
    },
    [],
  );
  async function copy() {
    try {
      await navigator.clipboard.writeText(code);
      setState("ok");
    } catch {
      setState("error");
    }
    if (timer.current) clearTimeout(timer.current);
    timer.current = setTimeout(() => setState(""), 2200);
  }
  return (
    <div className="code-block">
      <div className="code-toolbar">
        <span>
          <i />
          {label}
        </span>
        <button
          onClick={copy}
          aria-label={lang === "tr" ? "Kodu kopyala" : "Copy code"}
        >
          {state === "ok" ? <Check size={14} /> : <Copy size={14} />}
          <span>
            {state === "ok"
              ? lang === "tr"
                ? "Kopyalandı"
                : "Copied"
              : state === "error"
                ? lang === "tr"
                  ? "Seçip kopyalayın"
                  : "Select to copy"
                : lang === "tr"
                  ? "Kopyala"
                  : "Copy"}
          </span>
        </button>
      </div>
      <pre>
        <code>
          {code.split("\n").map((line, i) => (
            <span className="code-line" key={i}>
              <span className="line-no" aria-hidden="true">
                {i + 1}
              </span>
              <span
                className={
                  line.trim().startsWith("//") || line.trim().startsWith("# ")
                    ? "code-comment"
                    : ""
                }
              >
                {line
                  .split(
                    /("[^"\n]*"|\b(?:auto|int|float|return|class|public|private|const|void|using|namespace|override|for|if|true|false|include)\b|\b\d+(?:\.\d+)?f?\b)/g,
                  )
                  .map((p, j) => (
                    <span
                      key={j}
                      className={
                        /^"/.test(p)
                          ? "token-string"
                          : /^(auto|int|float|return|class|public|private|const|void|using|namespace|override|for|if|true|false|include)$/.test(
                                p,
                              )
                            ? "token-keyword"
                            : /^\d/.test(p)
                              ? "token-number"
                              : ""
                      }
                    >
                      {p}
                    </span>
                  ))}
              </span>
            </span>
          ))}
        </code>
      </pre>
    </div>
  );
}
export function Reveal({
  children,
  className = "",
}: {
  children: React.ReactNode;
  className?: string;
}) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const io = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          el.classList.add("revealed");
          io.disconnect();
        }
      },
      { threshold: 0.08 },
    );
    el.classList.add("reveal-ready");
    io.observe(el);
    return () => io.disconnect();
  }, []);
  return (
    <div ref={ref} className={`reveal ${className}`}>
      {children}
    </div>
  );
}
