import Link from "next/link";
export default function NotFound() {
  return (
    <main className="locale-entry">
      <span className="eyebrow">404 · OUT OF BOUNDS</span>
      <h1>
        Sayfa bulunamadı.
        <br />
        Page not found.
      </h1>
      <Link href="/tr/" className="button primary">
        NexusModel
      </Link>
    </main>
  );
}
