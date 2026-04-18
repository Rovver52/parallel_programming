import subprocess
import time

sizes = [200, 400, 800, 1200, 1600, 2000]
block_sizes = [8, 16, 32]

print("="*70)
print("CUDA BENCHMARK")
print("="*70)

for size in sizes:
    print(f"\n--- N={size} ---")
    for bs in block_sizes:
        cmd = f"./matrix_cuda matrix_{size}_A.txt matrix_{size}_B.txt {bs}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        for line in result.stdout.split('\n'):
            if "Execution time:" in line:
                print(f"  Block {bs}x{bs}: {line}")
            if "Performance:" in line:
                print(f"    {line}")
        time.sleep(0.5)