"""Generate readable reports, site CPU values and the guide's performance section.
Run after both benchmark scripts and benchmarks/results/optimized-modules.csv.
"""
import csv
import html
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT/'docs'
comparison = json.loads((DOCS/'lib/comparison-results.json').read_text(encoding='utf-8'))
revision = json.loads((DOCS/'lib/optimization-results.json').read_text(encoding='utf-8'))
modules = list(csv.DictReader((ROOT/'benchmarks/results/optimized-modules.csv').read_text(encoding='utf-8-sig').splitlines()))
metadata = comparison['metadata']
labels = ['Linear', 'Linear · scalar', 'Linear · SIMD', 'ReLU · scalar', 'ReLU · SIMD', 'Conv2D', 'MultiHeadAttention']
shapes = ['64 × 784 → 256', '32 × 256 → 256', '32 × 256 → 256', '1 × 2²⁰', '1 × 2²⁰', '8 × 16 × 28 × 28 · K32 · k3', '4 × 32 × 64 · H4']
cpu = [dict(op=label, shape=shape, ms=float(row['ms']), **{'pass':'forward' if 'ReLU' in label else 'forward + backward'})
       for label, shape, row in zip(labels, shapes, modules)]
assert len(cpu) == 7
(DOCS/'lib/cpu-results.json').write_text(json.dumps(cpu, indent=2, ensure_ascii=False), encoding='utf-8')
p = DOCS/'lib/benchmarks.ts'
gpu = p.read_text(encoding='utf-8').split('export const gpuRows =', 1)[1]
p.write_text('import cpu from "./cpu-results.json";\nexport const cpuRows = cpu;\n// Historical GPU snapshot; not remeasured for this revision.\nexport const gpuRows ='+gpu, encoding='utf-8')

report = ROOT/'benchmarks/RESULTS.md'
legacy = ROOT/'benchmarks/results/legacy-results.md'
if not legacy.exists():
    legacy.write_text(report.read_text(encoding='utf-8'), encoding='utf-8')
lines = ['# NexusModel CPU benchmark report', '', f"Measured: {metadata['measured_at']}", '',
         f"{metadata['cpu']} · {metadata['os']} · {metadata['compiler']}", '',
         'Float32, one compute thread. Three rotated rounds, 20 warmups per round, 30 samples per round, 5 calls per sample. Tables report medians of 90 sample means; p95 and all raw samples are in JSON.', '',
         '## Same-harness revision comparison', '', '| Operation | Before ms | After ms | Speedup |', '|---|---:|---:|---:|']
for row in revision['records']:
    lines.append(f"| {row['operation']} | {row['before_ms']:.6f} | {row['after_ms']:.6f} | {row['speedup']:.2f}× |")
lines += ['', 'Baseline rebuilt from the archived, digest-verified source snapshot. Same native harness for both revisions; backward includes dx/dw clearing. Revision order alternates. This compares the library revisions, not scalar versus SIMD.', '',
          '## NexusModel / NumPy / PyTorch', '', '| Operation / shape | NexusModel ms | NumPy ms | PyTorch ms | Lowest median |', '|---|---:|---:|---:|---|']
for row in comparison['records']:
    values = row['results']
    lines.append(f"| {row['operation']} {row['shape']} | "+' | '.join(f"{r['median_ms']:.6f}" for r in values)+f" | {min(values,key=lambda r:r['median_ms'])['library']} |")
wins = sum(min(row['results'],key=lambda r:r['median_ms'])['library']=='NexusModel' for row in comparison['records'])
lines += ['', f"NexusModel recorded the lowest median in {wins}/{len(comparison['records'])} measured workloads. This is a local observation, not a universal ranking or a statistical significance claim.", '',
          f"Versions: NumPy {comparison['records'][0]['results'][1]['version']}; PyTorch {comparison['records'][0]['results'][2]['version']}.", '',
          '## Full CPU modules', '', 'Current medians from 20 warmups and 30×5 calls (one round). Linear/Conv/MHA include forward and backward; ReLU forward only. These use the full module API with saved inputs. No valid pre-change module baseline was recorded.', '',
          '| Module | Shape | Median ms |', '|---|---|---:|']
for row in cpu:
    lines.append(f"| {row['op']} | {row['shape']} | {row['ms']:.6f} |")
lines += ['', '## Correctness and limits', '',
          '- Every cross-library output element was checked each round against float64 (rtol 1e-4, atol 1e-5); ReLU is exact.',
          '- Native tests independently cover seeded random matrices, tails, bias, optional gradients, repeated accumulation, grouped convolution, stable LayerNorm and tape growth.',
          '- NexusModel uses C++ calls; NumPy/PyTorch use Python APIs. Tiny-case timings include unequal dispatch overhead.',
          '- Preallocated outputs and warm caches; deterministic periodic benchmark inputs. Weight packing is timed. This is not full training or model throughput.',
          '- OS scheduling, affinity, thermals and power are uncontrolled. Close results can reverse between runs.',
          '- GPU numbers are historical and are not refreshed or mixed with current CPU comparisons.', '',
          '## Reproduce and raw evidence', '',
          'See `docs/benchmarks/README.md`. Machine metadata, loaded BLAS backends, package versions, source/binary/harness hashes and raw timing samples are in `docs/public/benchmark-comparison.json`. Revision evidence: `benchmarks/results/revision-comparison.json`. Archived sources: `benchmarks/results/baseline-source.zip`. Historical report: `benchmarks/results/legacy-results.md`.']
report.write_text('\n'.join(lines)+'\n', encoding='utf-8')

def table(headers, rows):
    return '<table><thead><tr>'+''.join('<th>'+html.escape(c)+'</th>' for c in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(c))+'</td>' for c in row)+'</tr>' for row in rows)+'</tbody></table>'

p = DOCS/'kilavuz.html'
guide = p.read_text(encoding='utf-8')
start = guide.index('<h1><span class="num">25</span>')
end = guide.index('<h1><span class="num">26</span>', start)
section = '<h1><span class="num">25</span>Performans ve optimizasyon</h1>'
section += f'<p>Ölçüm: {metadata["measured_at"][:10]}. {html.escape(metadata["cpu"])}. Float32, tek hesaplama iş parçacığı, Release.</p>'
section += '<p>Linear 6×16 register blokları ve 8 KiB ağırlık panelleri kullanır. Paketleme ölçüme dahildir. Geri geçiş geçici transpoz oluşturmaz; convolution aynı motoru paylaşır. LayerNorm iki geçişli float64 istatistik hesabını SIMD ile yürütür.</p><h2>Eski / yeni sürüm</h2>'
section += '<p>Arşivlenmiş kaynak ve yeni sürüm aynı C++ ölçüm koduyla derlendi. Üç turda sürüm sırası değiştirildi; her tur 20 ısınma ve 30×5 çağrı. Aşağıdaki değerler 90 örneğin medyanıdır.</p>'
section += table(['Çekirdek','Eski ms','Yeni ms','Oran'],[(r['operation'],f"{r['before_ms']:.4f}",f"{r['after_ms']:.4f}",f"{r['speedup']:.2f}×") for r in revision['records']])
section += '<h2>NumPy ve PyTorch karşılaştırması</h2><p>Üç turda kütüphane sırası değiştirilir. Aynı float32 girdiler, önceden ayrılmış çıktılar. NexusModel C++ çağrı maliyeti; NumPy/PyTorch Python çağrı maliyeti dahildir. Her çıktı float64 referansı ile doğrulanır. Süreler ms.</p>'
section += table(['İşlem / şekil','NexusModel','NumPy','PyTorch'],[(r['operation']+' '+r['shape'],*[f"{v['median_ms']:.5f}" for v in r['results']]) for r in comparison['records']])
section += f'<p>NexusModel {wins}/9 iş yükünde en düşük medyanı verdi. Küçük işlerde Python çağrı maliyeti önemlidir. Bu sonuçlar bütün boyutlarda veya bütün modellerde dünya birinciliği kanıtı değildir.</p>'
section += '<h2 style="break-before:page">Tam katman ölçümleri</h2><p>20 ısınma, 30×5 çağrı, tek tur medyanı. ReLU ileri; diğerleri ileri + geri. Eski katman sonuçlarıyla doğrudan optimizasyon oranı hesaplanmaz.</p>'
section += table(['Katman','Şekil','Medyan ms'],[(r['op'],r['shape'],f"{r['ms']:.5f}") for r in cpu])
section += '<div class="note"><strong>Ölçüm sınırları</strong>Güç profili, sıcaklık, çekirdek yakınlığı ve OS zamanlaması sabitlenmedi. GPU sonuçları eski sürüm arşividir ve bu optimizasyonda yeniden ölçülmedi. Ham örnekler: docs/public/benchmark-comparison.json ve optimization-results.json. Kaynaklar, sürümler ve hash değerleri raporda bulunur.</div>'
section += '<pre>cmake -S . -B build\ncmake --build build --config Release\nctest --test-dir build -C Release --output-on-failure\npython docs/benchmarks/compare_libraries.py</pre>\n'
guide = guide[:start]+section+guide[end:]
guide = guide.replace('-O3 -march=native -flto', '-O3; ISA bayrakları yalnız ilgili kaynak dosyalarında')
guide = guide.replace('src/core/kernels_nn.cpp','src/core/convolution.cpp, normalization.cpp, pooling.cpp, embedding.cpp, dropout.cpp')
guide = guide.replace('CPUID ve Windows’ta','CPUID ve').replace('AVX2 softmax: maksimum skalar bulunur','AVX2 softmax: maksimum SIMD ile bulunur')
guide = guide.replace('AVX-512’ye gider. Diğer sıcak','AVX-512 seçimine gider; Linear bu girişten bloklu AVX2 matris motoruna yönlenir. Diğer sıcak')
guide = guide.replace('<p><code>tests/test_model.cpp</code> tek çalıştırılabilirde toplanır:', '<p>Ek çekirdek testleri <code>tests/test_kernels.cpp</code> dosyasında; kuyruk boyutları, gradyan birikimi, gruplu convolution ve teyp büyümesini denetler. <code>tests/test_model.cpp</code> ise şu çalıştırılabilirde toplanır:')
guide = guide.replace('<tr><td>src/simd/</td><td>AVX2, AVX-512, dağıtım</td></tr>', '<tr><td>src/simd/avx2/</td><td>gemm.hpp, linear.cpp, normalization.cpp, details.txt</td></tr><tr><td>src/simd/</td><td>AVX2, AVX-512, atomik dağıtım</td></tr>')
p.write_text(guide, encoding='utf-8')
print('Updated report, site CPU data and HTML guide.')
