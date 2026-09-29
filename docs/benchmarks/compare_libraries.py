"""Reproduce CPU forward-only comparisons. Run from docs after building examples.
All measured APIs use preallocated outputs, float32 inputs and a single compute thread.
Python API dispatch is included for NumPy/PyTorch; C++ call overhead for NexusModel.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import csv, datetime, hashlib, json, platform, statistics, subprocess, time, re
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits,threadpool_info
ROOT=Path(__file__).resolve().parents[1]
os.chdir(ROOT)
Path('benchmark-data').mkdir(exist_ok=True)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
WARMUP,SAMPLES,ITERATIONS=20,30,5
ROUNDS=3

def cpu_name():
    if platform.system()=='Windows':
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
            return winreg.QueryValueEx(key,'ProcessorNameString')[0].strip()
    return platform.processor() or platform.machine()

def compiler_name():
    files=list(Path('.example-build/CMakeFiles').glob('*/CMakeCXXCompiler.cmake'))
    text=files[0].read_text(encoding='utf-8') if files else ''
    identity=re.search(r'set\(CMAKE_CXX_COMPILER_ID "([^"]+)"\)',text)
    version=re.search(r'set\(CMAKE_CXX_COMPILER_VERSION "([^"]+)"\)',text)
    return f'{identity[1] if identity else "unknown"} {version[1] if version else "unknown"} / Release' + (' /O2' if identity and identity[1]=='MSVC' else '')

def pattern(shape,mult,offset):
    values=((np.arange(np.prod(shape),dtype=np.int64)%101)*mult+offset)%101-50
    return (values.astype(np.float32)/np.float32(100)).reshape(shape)

def measure(fn):
    for _ in range(WARMUP):fn()
    values=[]
    for _ in range(SAMPLES):
        start=time.perf_counter_ns()
        for _ in range(ITERATIONS):fn()
        values.append((time.perf_counter_ns()-start)/1e6/ITERATIONS)
    return values

def summary(name,version,values,output,reference):
    error=float(np.max(np.abs(output.astype(np.float64)-reference)))
    valid=bool(np.allclose(output,reference,rtol=1e-4,atol=1e-5))
    if not valid:raise RuntimeError(f'{name} failed correctness: {error}')
    return dict(library=name,version=version,median_ms=statistics.median(values),p95_ms=float(np.percentile(values,95)),min_ms=min(values),max_abs_error=error,correct=True,samples_ms=values)

def native_case(exe,key):
    native=subprocess.run([str(exe),key],check=True,capture_output=True,text=True)
    row=json.loads(native.stdout)
    if row['id']!=key or len(row['samples'])!=SAMPLES:raise RuntimeError('Native benchmark protocol mismatch')
    return row['samples']

def rotated(exe,key,numpy_fn,torch_fn,reference,y,ty):
    samples=[[],[],[]]
    calls=[lambda:native_case(exe,key),lambda:measure(numpy_fn),lambda:measure(torch_fn)]
    for round_index in range(ROUNDS):
        for index in [(round_index+i)%3 for i in range(3)]:samples[index].extend(calls[index]())
        nx=np.fromfile(f'benchmark-data/{key}.bin',dtype=np.float32).reshape(reference.shape)
        # Validate every framework after every round, before any publication.
        for name,output in [('NexusModel',nx),('NumPy',y),('PyTorch',ty.numpy())]:
            if not np.allclose(output,reference,rtol=1e-4,atol=1e-5):raise RuntimeError(f'{name} failed correctness')
    return [summary('NexusModel','1.0.0',samples[0],nx,reference),summary('NumPy',np.__version__,samples[1],y,reference),summary('PyTorch',torch.__version__,samples[2],ty.numpy(),reference)]

with threadpool_limits(limits=1),torch.inference_mode():
    suffix='.exe' if platform.system()=='Windows' else ''
    exe=next((p for p in [ROOT/f'.example-build/Release/docs_compare_cpu{suffix}',ROOT/f'.example-build/docs_compare_cpu{suffix}'] if p.exists()),None)
    if exe is None:raise RuntimeError('Build docs/examples in Release first')
    records=[]
    for m,k,n in [(1,256,256),(7,129,33),(32,256,256),(96,768,192),(128,512,512),(256,1024,1024)]:
        key=f'gemm_{m}_{k}_{n}'
        x,w=pattern((m,k),17,3),pattern((n,k),13,7)
        wt=w.T
        y=np.empty((m,n),dtype=np.float32)
        tx,tw=torch.from_numpy(x),torch.from_numpy(w).T
        ty=torch.empty((m,n),dtype=torch.float32)
        reference=x.astype(np.float64)@w.astype(np.float64).T
        results=rotated(exe,key,lambda:np.matmul(x,wt,out=y),lambda:torch.mm(tx,tw,out=ty),reference,y,ty)
        records.append(dict(id=key,operation='GEMM',shape=f'{m} × {k} × {n}',definition='Y = X Wᵀ; no bias',results=results))
        print(key,[(r['library'],round(r['median_ms'],6)) for r in results],flush=True)
    for size in [1<<16,1<<20,1<<22]:
        key=f'relu_{size}';x=pattern((size,),17,3);y=np.empty_like(x);tx=torch.from_numpy(x);ty=torch.empty_like(tx)
        reference=np.maximum(x.astype(np.float64),0)
        results=rotated(exe,key,lambda:np.maximum(x,np.float32(0),out=y),lambda:torch.clamp(tx,min=0,out=ty),reference,y,ty)
        records.append(dict(id=key,operation='ReLU',shape=f'{size:,}',definition='Y = max(X, 0)',results=results))
        print(key,[(r['library'],round(r['median_ms'],6)) for r in results],flush=True)
    pools=[{k:v for k,v in info.items() if k!='filepath'} for info in threadpool_info()]
source_files=sorted((ROOT.parent/'include').rglob('*.hpp'))+sorted((ROOT.parent/'src').rglob('*.cpp'))+sorted((ROOT.parent/'src').rglob('*.hpp'))+[ROOT.parent/'CMakeLists.txt']
source_hash=hashlib.sha256(b''.join(f.relative_to(ROOT.parent).as_posix().encode()+b'\0'+f.read_bytes() for f in source_files)).hexdigest()
result=dict(metadata=dict(measured_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),cpu=cpu_name(),os=platform.platform(),python=platform.python_version(),compiler=compiler_name(),dtype='float32',threads=1,warmup=WARMUP,samples=SAMPLES*ROUNDS,rounds=ROUNDS,iterations_per_sample=ITERATIONS,timing='median and p95 of 90 samples from 3 rotated rounds, each the mean of 5 calls',scope='CPU forward-only, preallocated outputs; API dispatch included; weight packing included; no transfer, tensor allocation, autograd or backward',limitations='One machine, 3 rounds with rotated framework order; warm-cache steady state; deterministic periodic inputs. OS scheduling, affinity and thermal state uncontrolled. C++ and Python API overhead differ. NumPy is a numerical array library, not an end-to-end neural network framework.',blas=pools,torch_threads=torch.get_num_threads(),torch_interop_threads=torch.get_num_interop_threads(),nexus_source_sha256=source_hash,binary_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),harness_sha256=hashlib.sha256(Path(__file__).read_bytes()+Path('benchmarks/compare_cpu.cpp').read_bytes()).hexdigest(),correctness='Each round: compared every output element against float64 reference; rtol=1e-4, atol=1e-5; ReLU exact.'),records=records)
for target in ['lib/comparison-results.json','public/benchmark-comparison.json']:
    Path(target).write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
with open('public/benchmark-comparison.csv','w',newline='',encoding='utf-8-sig') as f:
    writer=csv.writer(f);writer.writerow(['operation','shape','library','version','median_ms','p95_ms','max_abs_error','threads','dtype'])
    for row in records:
        for r in row['results']:writer.writerow([row['operation'],row['shape'],r['library'],r['version'],r['median_ms'],r['p95_ms'],r['max_abs_error'],1,'float32'])
Path('benchmark-data/torch-config.txt').write_text(torch.__config__.show(),encoding='utf-8')
for row in records:
    print(row['operation'],row['shape'],[(r['library'],round(r['median_ms'],5)) for r in row['results']])
print('Correctness verified for every output. Results saved.')

