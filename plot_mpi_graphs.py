import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# ============================================
# ДАННЫЕ СТРОГО ИЗ СКРИНШОТОВ
# ============================================
experimental_data = [
    # 400x400
    {'size': 400, 'processes': 1, 'time_ms': 701.015},
    {'size': 400, 'processes': 2, 'time_ms': 351.35},
    {'size': 400, 'processes': 4, 'time_ms': 177.198},
    {'size': 400, 'processes': 8, 'time_ms': 89.869},

    # 1000x1000
    {'size': 1000, 'processes': 1, 'time_ms': 9838.09},
    {'size': 1000, 'processes': 2, 'time_ms': 4965.55},
    {'size': 1000, 'processes': 4, 'time_ms': 3207.71},
    {'size': 1000, 'processes': 8, 'time_ms': 1732.89},

    # 1600x1600
    {'size': 1600, 'processes': 1, 'time_ms': 57606.3},
    {'size': 1600, 'processes': 2, 'time_ms': 28788.6},
    {'size': 1600, 'processes': 4, 'time_ms': 14987.8},
    {'size': 1600, 'processes': 8, 'time_ms': 7626.48},

    # 2000x2000
    {'size': 2000, 'processes': 1, 'time_ms': 111498},
    {'size': 2000, 'processes': 2, 'time_ms': 55973.9},
    {'size': 2000, 'processes': 4, 'time_ms': 28269.5},
    {'size': 2000, 'processes': 8, 'time_ms': 14470.4},
]


def calculate_speedup(data):
    """Рассчитывает ускорение"""
    by_size = defaultdict(list)
    for entry in data:
        by_size[entry['size']].append(entry)

    baseline_times = {}
    for size, entries in by_size.items():
        one_proc_entries = [e for e in entries if e['processes'] == 1]
        if one_proc_entries:
            baseline_times[size] = one_proc_entries[0]['time_ms']

    results_with_speedup = []
    for entry in data:
        size = entry['size']
        processes = entry['processes']
        time_ms = entry['time_ms']

        if size in baseline_times:
            if processes == 1:
                speedup = 1.0
            else:
                speedup = baseline_times[size] / time_ms
        else:
            speedup = None

        results_with_speedup.append({
            **entry,
            'speedup': speedup
        })

    return results_with_speedup


def plot_time_vs_size(data):
    """График 1: Время от размера матрицы"""
    plt.figure(figsize=(12, 7))

    by_processes = defaultdict(list)
    for entry in data:
        by_processes[entry['processes']].append(entry)

    for processes in by_processes:
        by_processes[processes].sort(key=lambda x: x['size'])

    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    markers = ['o', 's', '^', 'd']

    for idx, (processes, entries) in enumerate(sorted(by_processes.items())):
        sizes = [e['size'] for e in entries]
        times = [e['time_ms'] / 1000 for e in entries]

        color_idx = idx % len(colors)
        marker_idx = idx % len(markers)

        plt.plot(sizes, times,
                 marker=markers[marker_idx],
                 color=colors[color_idx],
                 linewidth=2,
                 markersize=8,
                 label=f'{processes} процесс(ов)')

    plt.xlabel('Размер матрицы (N×N)', fontsize=12)
    plt.ylabel('Время выполнения (сек)', fontsize=12)
    plt.title('Зависимость времени выполнения от размера матрицы (MPI)', fontsize=14, pad=20)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(title='Количество процессов', loc='upper left')
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('mpi_time_vs_size.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ График 1 сохранён: mpi_time_vs_size.png")


def plot_speedup_vs_processes(data):
    """График 2: Ускорение от количества процессов"""
    plt.figure(figsize=(12, 7))

    data_with_speedup = calculate_speedup(data)

    by_size = defaultdict(list)
    for entry in data_with_speedup:
        if entry['speedup'] is not None:
            by_size[entry['size']].append(entry)

    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    markers = ['o', 's', '^', 'd']

    for idx, (size, entries) in enumerate(sorted(by_size.items())):
        processes = [e['processes'] for e in entries]
        speedups = [e['speedup'] for e in entries]

        sorted_data = sorted(zip(processes, speedups))
        processes = [p[0] for p in sorted_data]
        speedups = [p[1] for p in sorted_data]

        color_idx = idx % len(colors)
        marker_idx = idx % len(markers)

        plt.plot(processes, speedups,
                 marker=markers[marker_idx],
                 color=colors[color_idx],
                 linewidth=2,
                 markersize=8,
                 label=f'{size}×{size}')

    max_processes = max([e['processes'] for e in data])
    ideal_speedup = list(range(1, max_processes + 1))
    plt.plot(ideal_speedup, ideal_speedup,
             'k--', alpha=0.5, linewidth=1.5,
             label='Идеальное ускорение (линейное)')

    plt.xlabel('Количество процессов MPI', fontsize=12)
    plt.ylabel('Ускорение (S = T₁ / Tₚ)', fontsize=12)
    plt.title('Зависимость ускорения от количества процессов (MPI)', fontsize=14, pad=20)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(title='Размер матрицы', loc='upper left')
    plt.xticks([1, 2, 4, 8])
    plt.xlim(0.8, max_processes + 0.5)
    plt.tight_layout()
    plt.savefig('mpi_speedup_vs_processes.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ График 2 сохранён: mpi_speedup_vs_processes.png")


def print_statistics(data):
    """Вывод статистики"""
    print("\n" + "=" * 70)
    print("📊 СТАТИСТИКА ЭКСПЕРИМЕНТОВ (ДАННЫЕ ИЗ СКРИНШОТОВ)")
    print("=" * 70)

    data_with_speedup = calculate_speedup(data)

    by_size = defaultdict(list)
    for entry in data_with_speedup:
        by_size[entry['size']].append(entry)

    for size in sorted(by_size.keys()):
        entries = by_size[size]
        print(f"\n🔹 Матрица {size}×{size}:")

        for entry in sorted(entries, key=lambda x: x['processes']):
            time_sec = entry['time_ms'] / 1000
            processes = entry['processes']
            speedup = entry['speedup']

            if speedup is not None:
                print(f"   • {processes} процесс(ов): {time_sec:.4f} сек, ускорение: {speedup:.2f}×")

    print("\n" + "=" * 70)
    print(f"Всего экспериментов: {len(data)}")
    print("=" * 70)


def main():
    print("🎨 Построение графиков для лабораторной работы №3 (MPI)\n")
    print(f"📈 Найдено {len(experimental_data)} экспериментов из скриншотов\n")

    print_statistics(experimental_data)

    print("\n🔨 Построение графиков...")
    plot_time_vs_size(experimental_data)
    plot_speedup_vs_processes(experimental_data)

    print("\n✅ Все графики построены и сохранены!")


if __name__ == "__main__":
    main()