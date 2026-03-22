import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import rcParams

# Настройка стиля графиков
rcParams['font.family'] = 'DejaVu Sans'  # или 'Arial', 'Times New Roman'
rcParams['font.size'] = 11
rcParams['axes.labelsize'] = 12
rcParams['axes.titlesize'] = 14
rcParams['legend.fontsize'] = 10
rcParams['figure.figsize'] = (10, 6)
rcParams['savefig.dpi'] = 300
rcParams['savefig.bbox'] = 'tight'

# Цветовая схема
COLORS = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']


def generate_sample_data():
    """Генерация реалистичных тестовых данных (если нет реальных)"""
    np.random.seed(42)
    data = []

    sizes = [200, 400, 800, 1200, 1600, 2000]
    threads_list = [1, 2, 4, 8]

    for size in sizes:
        # Базовое время для 1 потока: ~2*N^3 операций, ~1 GFLOPS на ядро
        base_time = (2 * size ** 3) / (1e9 * 0.8)  # ~0.8 GFLOPS для последовательного кода

        for threads in threads_list:
            # Моделируем ускорение с учётом закона Амдала и накладных расходов
            # S(p) = 1 / ((1-P) + P/p + overhead)
            parallel_fraction = 0.95  # 95% кода можно распараллелить
            overhead = 0.02 * threads  # накладные расходы растут с числом потоков

            if threads == 1:
                speedup = 1.0
            else:
                serial_part = 1 - parallel_fraction
                parallel_part = parallel_fraction / threads
                speedup = 1 / (serial_part + parallel_part + overhead)
                speedup = min(speedup, threads * 0.95)  # ограничение идеального ускорения

            time_sec = base_time / speedup
            flops = 2 * size ** 3
            gflops = (flops / 1e9) / time_sec
            efficiency = (speedup / threads) * 100

            data.append({
                'Размер': size,
                'Потоки': threads,
                'Время(сек)': time_sec,
                'FLOPS': flops,
                'GFLOPS': gflops,
                'Ускорение': speedup,
                'Эффективность(%)': efficiency
            })

    return pd.DataFrame(data)


def load_or_generate_data(csv_file='openmp_statistics.csv'):
    """Загрузка данных из CSV или генерация тестовых"""
    if os.path.exists(csv_file):
        print(f"📁 Загрузка данных из {csv_file}...")
        try:
            df = pd.read_csv(csv_file, delimiter=';')
            # Приведение имён колонок к ожидаемым
            df.columns = [c.strip() for c in df.columns]
            print(f"✓ Загружено {len(df)} записей")
            return df
        except Exception as e:
            print(f"⚠️ Ошибка чтения файла: {e}")
            print("🔄 Используются тестовые данные...")
    else:
        print(f"⚠️ Файл {csv_file} не найден. Генерация тестовых данных...")

    return generate_sample_data()


def plot_time_vs_threads(df, output='time_vs_threads.png'):
    """График: время выполнения от количества потоков"""
    plt.figure()

    sizes = sorted(df['Размер'].unique())

    for i, size in enumerate(sizes):
        subset = df[df['Размер'] == size].sort_values('Потоки')
        plt.plot(subset['Потоки'], subset['Время(сек)'],
                 marker='o', label=f'{size}×{size}',
                 color=COLORS[i % len(COLORS)], linewidth=2)

    plt.xlabel('Количество потоков', fontsize=12)
    plt.ylabel('Время выполнения (сек)', fontsize=12)
    plt.title('Зависимость времени выполнения от количества потоков', fontsize=14, pad=20)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(title='Размер матрицы', loc='upper right')
    plt.xticks([1, 2, 4, 8])
    plt.yscale('log')  # Логарифмическая шкала для наглядности

    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📈 Сохранён график: {output}")


def plot_time_vs_size(df, output='time_vs_size.png'):
    """График: время выполнения от размера матрицы"""
    plt.figure()

    threads_list = sorted(df['Потоки'].unique())

    for i, threads in enumerate(threads_list):
        subset = df[df['Потоки'] == threads].sort_values('Размер')
        plt.plot(subset['Размер'], subset['Время(сек)'],
                 marker='s', label=f'{threads} поток(ов)',
                 color=COLORS[i % len(COLORS)], linewidth=2)

    plt.xlabel('Размер матрицы (N×N)', fontsize=12)
    plt.ylabel('Время выполнения (сек)', fontsize=12)
    plt.title('Зависимость времени выполнения от размера матрицы', fontsize=14, pad=20)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(title='Количество потоков', loc='upper left')
    plt.xscale('log')
    plt.yscale('log')

    # Добавим линию теоретической сложности O(N³)
    sizes = np.array(sorted(df['Размер'].unique()))
    ref = sizes ** 3 / sizes[0] ** 3 * df[(df['Размер'] == sizes[0]) & (df['Потоки'] == 1)]['Время(сек)'].values[0]
    plt.plot(sizes, ref, 'k--', alpha=0.4, label='O(N³) — теория')

    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📈 Сохранён график: {output}")


def plot_efficiency(df, output='efficiency.png'):
    """График: эффективность распараллеливания"""
    plt.figure()

    sizes = sorted(df['Размер'].unique())

    for i, size in enumerate(sizes):
        subset = df[df['Размер'] == size].sort_values('Потоки')
        # Фильтруем только многопоточные запуски
        subset = subset[subset['Потоки'] > 1]
        if len(subset) == 0:
            continue
        plt.plot(subset['Потоки'], subset['Эффективность(%)'],
                 marker='^', label=f'{size}×{size}',
                 color=COLORS[i % len(COLORS)], linewidth=2)

    # Линия 100% эффективности
    plt.axhline(y=100, color='gray', linestyle=':', alpha=0.5, label='Идеальная эффективность')
    # Линия 50% для ориентира
    plt.axhline(y=50, color='gray', linestyle=':', alpha=0.3)

    plt.xlabel('Количество потоков', fontsize=12)
    plt.ylabel('Эффективность (%)', fontsize=12)
    plt.title('Эффективность распараллеливания по закону Амдала', fontsize=14, pad=20)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(title='Размер матрицы', loc='lower left')
    plt.xticks([1, 2, 4, 8])
    plt.ylim(0, 110)

    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📈 Сохранён график: {output}")


def plot_combined_speedup(df, output='speedup_analysis.png'):
    """Дополнительный график: ускорение vs потоки с линией идеального ускорения"""
    plt.figure(figsize=(12, 7))

    sizes = sorted(df['Размер'].unique())

    for i, size in enumerate(sizes):
        subset = df[df['Размер'] == size].sort_values('Потоки')
        plt.plot(subset['Потоки'], subset['Ускорение'],
                 marker='o', label=f'{size}×{size}',
                 color=COLORS[i % len(COLORS)], linewidth=2, markersize=6)

    # Линия линейного ускорения (идеал)
    threads = np.array([1, 2, 4, 8])
    plt.plot(threads, threads, 'k--', alpha=0.5, linewidth=1.5, label='Линейное ускорение (идеал)')

    plt.xlabel('Количество потоков', fontsize=12)
    plt.ylabel('Ускорение (S = T₁ / Tₚ)', fontsize=12)
    plt.title('Масштабируемость параллельного умножения матриц', fontsize=14, pad=20)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(title='Размер матрицы', loc='upper left')
    plt.xticks([1, 2, 4, 8])
    plt.xlim(0.8, 8.5)

    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📈 Сохранён график: {output}")


def print_summary(df):
    """Вывод краткой статистики"""
    print("\n" + "=" * 60)
    print("📊 КРАТКАЯ СТАТИСТИКА")
    print("=" * 60)

    for size in sorted(df['Размер'].unique()):
        subset = df[df['Размер'] == size]
        t1 = subset[subset['Потоки'] == 1]['Время(сек)'].values
        t8 = subset[subset['Потоки'] == 8]['Время(сек)'].values if 8 in subset['Потоки'].values else None

        if len(t1) > 0:
            print(f"\n🔹 Матрица {size}×{size}:")
            print(f"   • 1 поток:  {t1[0]:.4f} сек  ({subset[subset['Потоки'] == 1]['GFLOPS'].values[0]:.2f} GFLOPS)")
            if t8 is not None and len(t8) > 0:
                speedup = t1[0] / t8[0]
                eff = (speedup / 8) * 100
                print(
                    f"   • 8 потоков: {t8[0]:.4f} сек  ({subset[subset['Потоки'] == 8]['GFLOPS'].values[0]:.2f} GFLOPS)")
                print(f"   • Ускорение: {speedup:.2f}×, Эффективность: {eff:.1f}%")


def main():
    print("🎨 Генерация графиков для лабораторной работы №2 (OpenMP)\n")

    # Загрузка данных
    df = load_or_generate_data()

    # Построение графиков
    print("\n🔨 Построение графиков...")
    plot_time_vs_threads(df)
    plot_time_vs_size(df)
    plot_efficiency(df)
    plot_combined_speedup(df)  # Бонусный график

    # Вывод статистики
    print_summary(df)

    print("\n✅ Все графики сохранены в текущей директории:")
    print("   • time_vs_threads.png")
    print("   • time_vs_size.png")
    print("   • efficiency.png")
    print("   • speedup_analysis.png")
    print("\n💡 Совет: Для отчёта используйте векторный формат — добавьте `plt.savefig(..., format='pdf')`")


if __name__ == "__main__":
    main()