import os
import subprocess
import csv
import re
import sys
import time
import platform


def get_available_cores():
    """Возвращает количество доступных логических ядер"""
    return os.cpu_count() or 4


def run_test(size, num_threads, exe_name, sequential_time=None):
    """Запускает тест для матрицы заданного размера и количества потоков"""
    print(f"  Тест: {size}x{size}, потоки={num_threads}...", end=" ")

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

    # Запускаем программу с указанием количества потоков
    run_cmd = [exe_name, "matrix_a.txt", "matrix_b.txt", "result.txt", str(num_threads)]

    # Замер времени выполнения процесса
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
        threads_match = re.search(r"THREADS:\s*(\d+)", content)

        if time_match and flops_match and threads_match:
            time_ms = float(time_match.group(1))
            time_sec = time_ms / 1000.0
            flops = int(flops_match.group(1))
            threads = int(threads_match.group(1))

            # Расчет ускорения и эффективности
            speedup = sequential_time / time_sec if sequential_time and sequential_time > 0 else 1.0
            efficiency = (speedup / threads) * 100 if threads > 0 else 100.0
            gflops = (flops / 1e9) / time_sec if time_sec > 0 else 0

            return {
                'size': size,
                'threads': threads,
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
    thread_counts = [1, 2, 4, 8]  # Будет автоматически ограничено доступными ядрами

    max_cores = get_available_cores()
    thread_counts = [t for t in thread_counts if t <= max_cores]

    exe_name = "matrix_mult_omp.exe" if os.name == 'nt' else "./matrix_mult_omp"

    if not os.path.exists(exe_name):
        print(f"❌ Файл {exe_name} не найден! Скомпилируйте программу сначала.")
        return

    print(f"🖥️  Доступно ядер: {max_cores}")
    print(f"📊 Размеры матриц: {sizes}")
    print(f"🧵 Количество потоков: {thread_counts}\n")

    all_results = []
    sequential_baseline = {}  # Хранение времени для 1 потока как baseline

    # Сначала запускаем тесты с 1 потоком для baseline
    print("📏 Измерение последовательного выполнения (baseline)...")
    for size in sizes:
        result = run_test(size, 1, exe_name)
        if result:
            sequential_baseline[size] = result['time_sec']
            all_results.append(result)
            print(f"✓ {size}x{size}: {result['time_sec']:.4f} сек, {result['gflops']:.2f} GFLOPS")
        else:
            print(f"✗ {size}x{size}: ошибка")

    print("\n🚀 Запуск параллельных тестов...")
    # Запускаем тесты с разным количеством потоков (кроме 1, уже измерено)
    for threads in thread_counts:
        if threads == 1:
            continue
        print(f"\n--- Потоки: {threads} ---")
        for size in sizes:
            baseline = sequential_baseline.get(size)
            result = run_test(size, threads, exe_name, baseline)
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
        csv_file = "openmp_statistics.csv"
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow(["Размер", "Потоки", "Время(сек)", "FLOPS", "GFLOPS", "Ускорение", "Эффективность(%)"])
            for r in sorted(all_results, key=lambda x: (x['size'], x['threads'])):
                writer.writerow([
                    r['size'],
                    r['threads'],
                    f"{r['time_sec']:.7f}",
                    r['flops'],
                    f"{r['gflops']:.3f}",
                    f"{r['speedup']:.3f}",
                    f"{r['efficiency']:.2f}"
                ])
        print(f"\n📁 Результаты сохранены в {csv_file}")

        # Генерация сводной таблицы по размерам
        print("\n📈 Сводная таблица ускорения:")
        print(f"{'Размер':<8} {'1 поток':<12}", end="")
        for t in thread_counts:
            if t > 1:
                print(f"{t} потоков:{'':<5}", end="")
        print()
        print("-" * (8 + 12 + len(thread_counts) * 15))

        for size in sizes:
            row_data = {r['threads']: r for r in all_results if r['size'] == size}
            if 1 in row_data:
                print(f"{size:<8} {row_data[1]['time_sec']:<12.4f}", end="")
                for t in thread_counts:
                    if t > 1 and t in row_data:
                        print(f"{row_data[t]['speedup']:.2f}x ({row_data[t]['efficiency']:.1f}%){'':<5}", end="")
                print()
    else:
        print("❌ Нет данных для сохранения!")

    # Очистка
    for f in ["matrix_a.txt", "matrix_b.txt", "result.txt"]:
        if os.path.exists(f):
            os.remove(f)


if __name__ == "__main__":
    main()