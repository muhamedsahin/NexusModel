# NexusModel documentation

A bilingual Next.js App Router documentation site with a Three.js tensor scene, KaTeX math, source snapshots, search, persistent themes, reduced-motion support, and published benchmark comparisons.

## Run

Requires Node.js 20.9+ and npm. From `docs`:

```sh
npm ci
npm run dev
```

Development: http://127.0.0.1:3000/tr/ (English: `/en/`).

Production:

```sh
npm run build
npm start
```

`build` synchronizes public source snapshots and exports the complete site into `out/`. `start` serves that static export on localhost:3000. Use `npm start -- --port 3100` for another port. The output can be hosted by any static server at the domain root.

## Content and design

- `DESIGN.md`: visual direction, narrative, and interaction decisions.
- `lib/content-basics.ts`: getting started, tensor, modules, backward, and containers.
- `lib/content-layers.ts`: all layer families, formulas, defaults, and limitations.
- `lib/content-systems.ts`: initialization, serialization, SIMD, CUDA, testing, methodology.
- `lib/benchmarks.ts`: published values from `../benchmarks/RESULTS.md`.
- `scripts/sync-sources.mjs`: refreshes original-source downloads during the build.
- `app/globals.css`: responsive dark/light visual system.

Both languages are stored together for each article and section. New articles are exported automatically and appear in navigation/search. Benchmark values are recorded results, not live browser measurements. GPU figures exclude host/device transfers. CPU/compiler metadata is recorded with the current measurements; GPU numbers remain an explicitly labeled historical snapshot.

`kilavuz.html` and `NexusModel-Kullanim-Kilavuzu.pdf` include the current performance report. Regenerate using `python benchmarks/export_reports.py` then `node scripts/render-guide.mjs`.

## Verification

```sh
npm run typecheck
npm run build
npm start -- --port 3100
# In another terminal (install Playwright Chromium if needed):
npx playwright install chromium
npm test
```

Tests cover desktop/mobile layouts, WebGL availability, language/theme persistence, search, benchmark filters/downloads, reduced motion, route generation and source links. Screenshots are saved under `shots/`.

The hero is a conceptual tensor visualization, not a profiler. The loading reveal is a short brand animation and does not display fake progress. WebGL has a static fallback; offscreen/hidden scenes stop rendering. Fonts are bundled locally with their package licenses.

## Cross-library comparison

The benchmark page includes a separate, dated local CPU study of NexusModel, NumPy, and PyTorch. It is distinct from the repository's historical CPU/GPU report. See `benchmarks/README.md` for reproduction. The site serves recorded results and never pretends to run native benchmarks in the browser.

## Executable documentation examples

```sh
node scripts/extract-examples.mjs
cmake -S examples -B .example-build
cmake --build .example-build --config Release
ctest --test-dir .example-build -C Release --output-on-failure
```

`quickstart.cpp` verifies the model's shapes; `train_mse.cpp` learns y=2x+1 and verifies the final loss. `scripts/verify-examples.mjs` also normalizes duplicate case-insensitive Windows environment keys before invoking MSBuild.
