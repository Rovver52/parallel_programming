import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import rcParams

rcParams['font.family'] = 'DejaVu Sans'
rcParams['font.size'] = 11
rcParams['figure.figsize'] = (10, 6)
rcParams['savefig.dpi'] = 300
rcParams['savefig.bbox'] = 'tight'

COLORS = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']


def generate_sample_data():
    """Генерация реалистичных тестовых данных"""
    np.random.seed(42)
    data = []

    sizes = [200, 400, 800, 1200, 1600, 2000]
    processes_list = [1, 2, 4, 8]

    for size in sizes:
        base_time = (2 * size ** 3) / (1e9 * 0.7)  # ~0.7 GFLOPS для последовательного MPI

        for processes in processes_list:
            parallel_fraction = 0.93  # MPI имеет больше накладных расходов чем OpenMP
            overhead = 0.03 * processes  # коммуникационные расходы

            if processes == 1:
                speedup = 1.0
            else:
                serial_part = 1 - parallel_fraction
                parallel_part = parallel_fraction / processes
                speedup = 1 / (serial_part + parallel_part + overhead)
                speedup = min(speedup, processes * 0.90)

            time_sec = base_time / speedup
            flops = 2 * size ** 3
            gflops = (flops / 1e9) / time_sec
            efficiency = (speedup / processes) * 100

            data.append({
                'Размер': size,
                'Процессы': processes,
                'Время(сек)': time_sec,
                'FLOPS': flops,
                'GFLOPS': gflops,
                'Ускорение': speedup,
                'Эффективность(%)': efficiency
            })

    return pd.DataFrame(data)


def load_or_generate_data(csv_file='mpi_statistics.csv'):
    if os.path.exists(csv_file):
        print(f"📁 Загрузка данных из {csv_file}...")
        try:
            df = pd.read_csv(csv_file, delimiter=';')
            df.columns = [c.strip() for c in df.columns]
            print(f"✓ Загружено {len(df)} записей")
            return df
        except Exception as e:
            print(f"⚠️ Ошибка чтения файла: {e}")
    else:
        print(f"⚠️ Файл {csv_file} не найден. Генерация тестовых данных...")

    return generate_sample_data()


def plot_time_vs_processes(df, output='mpi_time_vs_processes.png'):
    plt.figure()

    sizes = sorted(df['Размер'].unique())

    for i, size in enumerate(sizes):
        subset = df[df['Размер'] == size].sort_values('Процессы')
        plt.plot(subset['Процессы'], subset['Время(сек)'],
                 marker='o', label=f'{size}×{size}',
                 color=COLORS[i % len(COLORS)], linewidth=2)

    plt.xlabel('Количество процессов MPI', fontsize=12)
    plt.ylabel('Время выполнения (сек)', fontsize=12)
    plt.title('Зависимость времени выполнения от количества процессов (MPI)', fontsize=14, pad=20)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(title='Размер матрицы', loc='upper right')
    plt.xticks([1, 2, 4, 8])
    plt.yscale('log')

    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📈 Сохранён график: {output}")


def plot_time_vs_size(df, output='mpi_time_vs_size.png'):
    plt.figure()

    processes_list = sorted(df['Процессы'].unique())

    for i, processes in enumerate(processes_list):
        subset = df[df['Процессы'] == processes].sort_values('Размер')
        plt.plot(subset['Размер'], subset['Время(сек)'],
                 marker='s', label=f'{processes} процесс(ов)',
                 color=COLORS[i % len(COLORS)], linewidth=2)

    plt.xlabel('Размер матрицы (N×N)', fontsize=12)
    plt.ylabel('Время выполнения (сек)', fontsize=12)
    plt.title('Зависимость времени выполнения от размера матрицы (MPI)', fontsize=14, pad=20)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(title='Количество процессов', loc='upper left')
    plt.xscale('log')
    plt.yscale('log')

    sizes = np.array(sorted(df['Размер'].unique()))
    ref = sizes ** 3 / sizes[0] ** 3 * df[(df['Размер'] == sizes[0]) & (df['Процессы'] == 1)]['Время(сек)'].values[0]
    plt.plot(sizes, ref, 'k--', alpha=0.4, label='O(N³) — теория')

    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📈 Сохранён график: {output}")


def plot_efficiency(df, output='mpi_efficiency.png'):
    plt.figure()

    sizes = sorted(df['Размер'].unique())

    for i, size in enumerate(sizes):
        subset = df[df['Размер'] == size].sort_values('Процессы')
        subset = subset[subset['Процессы'] > 1]
        if len(subset) == 0:
            continue
        plt.plot(subset['Процессы'], subset['Эффективность(%)'],
                 marker='^', label=f'{size}×{size}',
                 color=COLORS[i % len(COLORS)], linewidth=2)

    plt.axhline(y=100, color='gray', linestyle=':', alpha=0.5, label='Идеальная эффективность')
    plt.axhline(y=50, color='gray', linestyle=':', alpha=0.3)

    plt.xlabel('Количество процессов MPI', fontsize=12)
    plt.ylabel('Эффективность (%)', fontsize=12)
    plt.title('Эффективность распараллеливания (MPI)', fontsize=14, pad=20)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(title='Размер матрицы', loc='lower left')
    plt.xticks([1, 2, 4, 8])
    plt.ylim(0, 110)

    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📈 Сохранён график: {output}")


def plot_speedup(df, output='mpi_speedup.png'):
    plt.figure(figsize=(12, 7))

    sizes = sorted(df['Размер'].unique())

    for i, size in enumerate(sizes):
        subset = df[df['Размер'] == size].sort_values('Процессы')
        plt.plot(subset['Процессы'], subset['Ускорение'],
                 marker='o', label=f'{size}×{size}',
                 color=COLORS[i % len(COLORS)], linewidth=2, markersize=6)

    threads = np.array([1, 2, 4, 8])
    plt.plot(threads, threads, 'k--', alpha=0.5, linewidth=1.5, label='Линейное ускорение (идеал)')

    plt.xlabel('Количество процессов MPI', fontsize=12)
    plt.ylabel('Ускорение (S = T₁ / Tₚ)', fontsize=12)
    plt.title('Масштабируемость параллельного умножения матриц (MPI)', fontsize=14, pad=20)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(title='Размер матрицы', loc='upper left')
    plt.xticks([1, 2, 4, 8])
    plt.xlim(0.8, 8.5)

    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📈 Сохранён график: {output}")


def print_summary(df):
    print("\n" + "=" * 60)
    print("📊 КРАТКАЯ СТАТИСТИКА MPI")
    print("=" * 60)

    for size in sorted(df['Размер'].unique()):
        subset = df[df['Размер'] == size]
        t1 = subset[subset['Процессы'] == 1]['Время(сек)'].values
        t8 = subset[subset['Процессы'] == 8]['Время(сек)'].values if 8 in subset['Процессы'].values else None

        if len(t1) > 0:
            print(f"\n🔹 Матрица {size}×{size}:")
            print(
                f"   • 1 процесс:  {t1[0]:.4f} сек  ({subset[subset['Процессы'] == 1]['GFLOPS'].values[0]:.2f} GFLOPS)")
            if t8 is not None and len(t8) > 0:
                speedup = t1[0] / t8[0]
                eff = (speedup / 8) * 100
                print(
                    f"   • 8 процессов: {t8[0]:.4f} сек  ({subset[subset['Процессы'] == 8]['GFLOPS'].values[0]:.2f} GFLOPS)")
                print(f"   • Ускорение: {speedup:.2f}×, Эффективность: {eff:.1f}%")


def main():
    print("🎨 Генерация графиков для лабораторной работы №3 (MPI)\n")

    df = load_or_generate_data()

    print("\n🔨 Построение графиков...")
    plot_time_vs_processes(df)
    plot_time_vs_size(df)
    plot_efficiency(df)
    plot_speedup(df)

    print_summary(df)

    print("\n✅ Все графики сохранены:")
    print("   • mpi_time_vs_processes.png")
    print("   • mpi_time_vs_size.png")
    print("   • mpi_efficiency.png")
    print("   • mpi_speedup.png")


if __name__ == "__main__":
    main()