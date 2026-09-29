"""Compare the archived and optimized native kernels with identical timing code.
Rebuild baseline_probe from benchmarks/results/baseline-source.zip first.
Both executables use benchmarks/bench_kernels.cpp from the current checkout.
"""
import csv
import datetime
import hashlib
import io
import json
import statistics
import subprocess
import os
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if '--prepare' in sys.argv:
    source = ROOT/'benchmarks/.baseline/source'
    source.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ROOT/'benchmarks/results/baseline-source.zip') as archive:
        for name in archive.namelist():
            if not (source/name).resolve().is_relative_to(source.resolve()):
                raise RuntimeError('Invalid archive path')
        archive.extractall(source)
    cmake = source/'CMakeLists.txt'
    cmake.write_text(cmake.read_text(encoding='utf-8') + '\nadd_executable(baseline_probe "' +
        (ROOT/'benchmarks/bench_kernels.cpp').as_posix() + '")\ntarget_link_libraries(baseline_probe PRIVATE nexus_model)\n', encoding='utf-8')
    environment = {k.upper(): v for k,v in os.environ.items()} if os.name == 'nt' else os.environ.copy()
    subprocess.run(['cmake','-S',str(source),'-B',str(source.parent/'build'),'-DCMAKE_BUILD_TYPE=Release','-DNEXUS_MODEL_BUILD_TESTS=OFF'], env=environment, check=True)
    subprocess.run(['cmake','--build',str(source.parent/'build'),'--config','Release','--target','baseline_probe'], env=environment, check=True)
def executable(directory, name):
    name += '.exe' if os.name == 'nt' else ''
    return next(p for p in [directory/'Release'/name,directory/name] if p.exists())
executables = [executable(ROOT/'benchmarks/.baseline/build','baseline_probe'),
               executable(ROOT/'build','nexus_model_bench_kernels')]
samples = [{}, {}]
for round_index in range(3):
    for index in ([0, 1] if round_index % 2 == 0 else [1, 0]):
        result = subprocess.run([str(executables[index])], check=True, capture_output=True, text=True)
        for row in csv.DictReader(io.StringIO(result.stdout)):
            samples[index].setdefault(row['operation'], []).extend(float(v) for v in row['samples_ms'].split(';'))
    print(f'Completed revision round {round_index+1}/3', flush=True)
records = []
for operation in samples[0]:
    before, after = [statistics.median(s[operation]) for s in samples]
    records.append(dict(operation=operation, before_ms=before, after_ms=after, speedup=before/after,
                        before_samples_ms=samples[0][operation], after_samples_ms=samples[1][operation]))
result = dict(metadata=dict(measured_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    rounds=3, samples=90, warmup=20, iterations_per_sample=5,
    method='Same native timing harness, alternating revision order; median of 90 five-call averages. Backward includes clearing dx/dw buffers. No affinity, thermal or power control.',
    baseline_source_zip_sha256=hashlib.sha256((ROOT/'benchmarks/results/baseline-source.zip').read_bytes()).hexdigest(),
    binary_sha256=[hashlib.sha256(p.read_bytes()).hexdigest() for p in executables]), records=records)
for target in ['benchmarks/results/revision-comparison.json', 'docs/lib/optimization-results.json', 'docs/public/optimization-results.json']:
    (ROOT/target).write_text(json.dumps(result, indent=2), encoding='utf-8')
for row in records:
    print(row['operation'], round(row['before_ms'],6), '->', round(row['after_ms'],6), f"{row['speedup']:.2f}x", flush=True)
