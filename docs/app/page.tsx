import Link from "next/link";
export default function Page() {
  return (
    <main className="locale-entry">
      <div className="brand-mark">N</div>
      <h1>NexusModel</h1>
      <p>Computation, made visible.</p>
      <div className="actions">
        <Link className="button primary" href="/tr/">
          Türkçe
        </Link>
        <Link className="button secondary" href="/en/">
          English
        </Link>
      </div>
      <script
        dangerouslySetInnerHTML={{
          __html: `try{location.replace(localStorage.getItem('nx-locale')==='en'?'/en/':'/tr/')}catch(e){location.replace('/tr/')}`,
        }}
      />
    </main>
  );
}
