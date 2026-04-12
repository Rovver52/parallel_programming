import os
import subprocess
import csv
import re
import sys
import time


def get_available_cores():
    """Возвращает количество доступных логических ядер"""
    return os.cpu_count() or 4


def run_test(size, num_processes, exe_name, sequential_time=None):
    """Запускает тест для матрицы заданного размера и количества процессов"""
    print(f"  Тест: {size}x{size}, процессы={num_processes}...", end=" ")

    # Удаляем старые файлы
    for f in ["matrix_a.txt", "matrix_b.txt", "result.txt"]:
        if os.path.exists(f):
            os.remove(f)

    # Генерируем входные файлы
    generate_cmd = [sys.executable, "generate.py", str(size), "matrix_a.txt", "matrix_b.txt"]
    result = subprocess.run(generate_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Ошибка генерации")
        return None

    time.sleep(0.1)

    # Запускаем программу MPI с указанием количества процессов
    if os.name == 'nt':
        run_cmd = ["mpiexec", "-n", str(num_processes), exe_name,
                   "matrix_a.txt", "matrix_b.txt", "result.txt"]
    else:
        run_cmd = ["mpirun", "-np", str(num_processes), exe_name,
                   "matrix_a.txt", "matrix_b.txt", "result.txt"]

    proc_start = time.time()
    result = subprocess.run(run_cmd, capture_output=True, text=True)
    proc_time = time.time() - proc_start

    if result.returncode != 0:
        print(f"❌ Ошибка: {result.stderr[:100]}")
        return None

    time.sleep(0.1)

    # Читаем метаданные из result.txt
    try:
        with open("result.txt", "r", encoding="utf-8") as f:
            content = f.read()

        time_match = re.search(r"TIME_MS:\s*([\d.]+)", content)
        flops_match = re.search(r"FLOPS:\s*(\d+)", content)
        proc_match = re.search(r"PROCESSES:\s*(\d+)", content)

        if time_match and flops_match and proc_match:
            time_ms = float(time_match.group(1))
            time_sec = time_ms / 1000.0
            flops = int(flops_match.group(1))
            processes = int(proc_match.group(1))

            # Расчет ускорения и эффективности
            speedup = sequential_time / time_sec if sequential_time and sequential_time > 0 else 1.0
            efficiency = (speedup / processes) * 100 if processes > 0 else 100.0
            gflops = (flops / 1e9) / time_sec if time_sec > 0 else 0

            return {
                'size': size,
                'processes': processes,
                'time_sec': time_sec,
                'flops': flops,
                'gflops': gflops,
                'speedup': speedup,
                'efficiency': efficiency
            }
        else:
            print(f"❌ Нет метаданных")
            return None

    except Exception as e:
        print(f"❌ Ошибка чтения: {e}")
        return None


def main():
    # Конфигурация тестов
    sizes = [200, 400, 800, 1200, 1600, 2000]
    process_counts = [1, 2, 4, 8]

    max_cores = get_available_cores()
    process_counts = [p for p in process_counts if p <= max_cores]

    exe_name = "matrix_mult_mpi.exe" if os.name == 'nt' else "./matrix_mult_mpi"

    if not os.path.exists(exe_name):
        print(f"❌ Файл {exe_name} не найден! Скомпилируйте программу сначала.")
        print(f"   Команда: mpicxx -O3 -o matrix_mult_mpi matrix_mult_mpi.cpp")
        return

    print(f"🖥️  Доступно ядер: {max_cores}")
    print(f"📊 Размеры матриц: {sizes}")
    print(f"🔄 Количество процессов: {process_counts}\n")

    all_results = []
    sequential_baseline = {}

    # Сначала запускаем тесты с 1 процессом для baseline
    print("📏 Измерение последовательного выполнения (baseline)...")
    for size in sizes:
        result = run_test(size, 1, exe_name)
        if result:
            sequential_baseline[size] = result['time_sec']
            all_results.append(result)
            print(f"✓ {size}x{size}: {result['time_sec']:.4f} сек, {result['gflops']:.2f} GFLOPS")
        else:
            print(f"✗ {size}x{size}: ошибка")

    print("\n🚀 Запуск параллельных тестов MPI...")
    for processes in process_counts:
        if processes == 1:
            continue
        print(f"\n--- Процессы: {processes} ---")
        for size in sizes:
            baseline = sequential_baseline.get(size)
            result = run_test(size, processes, exe_name, baseline)
            if result:
                all_results.append(result)
                print(f"✓ {size}x{size}: {result['time_sec']:.4f} сек, "
                      f"{result['gflops']:.2f} GFLOPS, "
                      f"ускорение: {result['speedup']:.2f}x, "
                      f"эффективность: {result['efficiency']:.1f}%")
            else:
                print(f"✗ {size}x{size}: ошибка")

    # Сохранение результатов в CSV
    if all_results:
        csv_file = "mpi_statistics.csv"
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow(["Размер", "Процессы", "Время(сек)", "FLOPS", "GFLOPS", "Ускорение", "Эффективность(%)"])
            for r in sorted(all_results, key=lambda x: (x['size'], x['processes'])):
                writer.writerow([
                    r['size'],
                    r['processes'],
                    f"{r['time_sec']:.7f}",
                    r['flops'],
                    f"{r['gflops']:.3f}",
                    f"{r['speedup']:.3f}",
                    f"{r['efficiency']:.2f}"
                ])
        print(f"\n📁 Результаты сохранены в {csv_file}")

        # Генерация сводной таблицы
        print("\n📈 Сводная таблица ускорения:")
        print(f"{'Размер':<8} {'1 процесс':<12}", end="")
        for p in process_counts:
            if p > 1:
                print(f"{p} процессов:{'':<5}", end="")
        print()
        print("-" * (8 + 12 + len(process_counts) * 15))

        for size in sizes:
            row_data = {r['processes']: r for r in all_results if r['size'] == size}
            if 1 in row_data:
                print(f"{size:<8} {row_data[1]['time_sec']:<12.4f}", end="")
                for p in process_counts:
                    if p > 1 and p in row_data:
                        print(f"{row_data[p]['speedup']:.2f}x ({row_data[p]['efficiency']:.1f}%){'':<5}", end="")
                print()
    else:
        print("❌ Нет данных для сохранения!")

    # Очистка
    for f in ["matrix_a.txt", "matrix_b.txt", "result.txt"]:
        if os.path.exists(f):
            os.remove(f)


if __name__ == "__main__":
    main()