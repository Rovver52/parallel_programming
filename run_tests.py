import os
import subprocess
import csv
import re
import sys
import time


def run_test(size):
    """Запускает тест для матрицы заданного размера и возвращает статистику"""
    print(f"Тестирование матрицы размера {size}x{size}...")

    # Удаляем старые файлы
    for f in ["matrix_a.txt", "matrix_b.txt", "result.txt"]:
        if os.path.exists(f):
            os.remove(f)

    # Генерируем входные файлы
    print(f"  Генерация матриц...")
    generate_cmd = [sys.executable, "generate.py", str(size), "matrix_a.txt", "matrix_b.txt"]
    result = subprocess.run(generate_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  Ошибка генерации: {result.stderr}")
        return None

    # Небольшая пауза чтобы файлы точно записались
    time.sleep(0.1)

    # Запускаем программу на C++
    print(f"  Запуск умножения...")
    exe_name = "matrix_mult.exe" if os.name == 'nt' else "./matrix_mult"
    run_cmd = [exe_name, "matrix_a.txt", "matrix_b.txt", "result.txt"]
    result = subprocess.run(run_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  Ошибка выполнения: {result.stderr}")
        return None

    # Небольшая пауза чтобы файл точно записался
    time.sleep(0.1)

    # Читаем метаданные из result.txt
    try:
        with open("result.txt", "r", encoding="utf-8") as f:
            content = f.read()

        # Отладка: выводим часть файла
        # print(f"  Содержимое result.txt: {content[-200:]}")

        # Извлекаем данные из метаданных
        time_match = re.search(r"TIME_MS:\s*([\d.]+)", content)
        flops_match = re.search(r"FLOPS:\s*(\d+)", content)
        memory_match = re.search(r"MEMORY_BYTES:\s*(\d+)", content)

        if time_match and flops_match and memory_match:
            time_ms = float(time_match.group(1))
            time_sec = time_ms / 1000.0  # Переводим в секунды
            flops = int(flops_match.group(1))
            memory_bytes = int(memory_match.group(1))

            return {
                'size': size,
                'time_sec': time_sec,
                'flops': flops,
                'memory_bytes': memory_bytes
            }
        else:
            print(f"  Не удалось найти метаданные в файле!")
            print(f"  TIME_MS: {time_match}")
            print(f"  FLOPS: {flops_match}")
            print(f"  MEMORY_BYTES: {memory_match}")
            return None

    except Exception as e:
        print(f"Ошибка чтения result.txt: {e}")
        return None


def main():
    # Размеры матриц для тестирования
    sizes = [200, 400, 800, 1200, 1600, 2000]

    results = []

    print("Начало тестирования...\n")

    for size in sizes:
        result = run_test(size)
        if result:
            results.append(result)
            print(f"✓ Размер {size}: время={result['time_sec']:.6f} сек, "
                  f"операций={result['flops']}, память={result['memory_bytes']} байт\n")
        else:
            print(f"✗ Размер {size}: ошибка\n")

    # Записываем результаты в CSV
    if results:
        csv_file = "matrix_statistics.csv"
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=";")

            # Заголовок
            writer.writerow(["Размер матрицы", "Время выполнения(сек.)",
                             "Количество операций", "Объём данных"])

            # Данные
            for r in results:
                writer.writerow([
                    r['size'],
                    f"{r['time_sec']:.7f}",
                    r['flops'],
                    r['memory_bytes']
                ])

        print(f"Результаты сохранены в файл {csv_file}")
    else:
        print("Нет данных для сохранения!")

    # Очищаем временные файлы
    for file in ["matrix_a.txt", "matrix_b.txt", "result.txt"]:
        if os.path.exists(file):
            os.remove(file)


if __name__ == "__main__":
    main()