import matplotlib.pyplot as plt
import numpy as np

# Настройка стиля
plt.style.use('seaborn-v0_8-darkgrid')

# Создаем фигуру с двумя подграфиками
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Данные
sizes = [200, 400, 800, 1200, 1600, 2000]
cpu_times = [0.2165, 1.7656, 10.2694, 36.1968, 88.5648, 315.82]
gpu_times = [0.17773, 0.176111, 0.2346, 0.290668, 0.354897, 0.477212]

# =============================================
# ГРАФИК 1: Сравнение времени выполнения CPU vs GPU
# =============================================
ax1 = axes[0]
ax1.plot(sizes, cpu_times, 's--', label='CPU (AMD Ryzen 5 4600H)', linewidth=2, color='red')
ax1.plot(sizes, gpu_times, 'o-', label='GPU (NVIDIA Tesla T4)', linewidth=2, color='green')
ax1.set_xlabel('Размер матрицы N', fontsize=12)
ax1.set_ylabel('Время выполнения (секунды)', fontsize=12)
ax1.set_title('Сравнение времени выполнения: CPU vs GPU', fontsize=14)
ax1.set_yscale('log')
ax1.legend()
ax1.grid(True, alpha=0.3)

# =============================================
# ГРАФИК 2: Ускорение GPU относительно CPU
# =============================================
ax2 = axes[1]
speedups = [cpu/gpu for cpu, gpu in zip(cpu_times, gpu_times)]
ax2.plot(sizes, speedups, 'o-', linewidth=2, color='purple', markersize=8)
ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='Базовый уровень (1x)')
ax2.set_xlabel('Размер матрицы N', fontsize=12)
ax2.set_ylabel('Ускорение (CPU/GPU)', fontsize=12)
ax2.set_title('Ускорение GPU относительно CPU', fontsize=14)
ax2.grid(True, alpha=0.3)
ax2.legend()

# Добавляем подписи значений ускорения над точками
for size, sp in zip(sizes, speedups):
    ax2.annotate(f'{sp:.1f}x', (size, sp), textcoords="offset points", xytext=(0, 10), ha='center')

# Настраиваем отступы
plt.tight_layout()

# Сохраняем график
plt.savefig('cuda_speedup.png', dpi=150, bbox_inches='tight')
plt.show()

# Выводим максимальное ускорение
print(f"\nМаксимальное ускорение: {max(speedups):.1f}x при размере N={sizes[speedups.index(max(speedups))]}")
