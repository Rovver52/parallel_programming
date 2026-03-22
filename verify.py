import sys
import numpy as np


def read_matrix_cpp(filename):
    """Читает матрицу из файла, созданного C++ программой."""
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()

        # Находим строку с размером
        n = int(lines[0].strip())

        matrix = []
        for i in range(1, n + 1):
            row = [float(x) for x in lines[i].split()]
            matrix.append(row)

        return np.array(matrix), n
    except Exception as e:
        print(f"Ошибка чтения файла {filename}: {e}")
        return None, None


def read_input_matrix(filename):
    """Читает входную матрицу (простой формат: N затем данные)."""
    try:
        with open(filename, 'r') as f:
            content = f.read().split()

        iterator = iter(content)
        n = int(next(iterator))
        matrix = []
        for _ in range(n):
            row = []
            for _ in range(n):
                row.append(float(next(iterator)))
            matrix.append(row)
        return np.array(matrix)
    except Exception as e:
        print(f"Ошибка чтения входного файла {filename}: {e}")
        return None


def main():
    if len(sys.argv) != 4:
        print("Использование: python verify.py <файл_матрицы_A> <файл_матрицы_B> <файл_результата>")
        sys.exit(1)

    file_a = sys.argv[1]
    file_b = sys.argv[2]
    file_res = sys.argv[3]

    print("--- Верификация результатов ---")

    # Чтение входных данных
    A = read_input_matrix(file_a)
    B = read_input_matrix(file_b)
    C_cpp, n = read_matrix_cpp(file_res)

    if A is None or B is None or C_cpp is None:
        print("ВЕРИФИКАЦИЯ: ПРОВАЛ (Ошибка чтения файлов)")
        sys.exit(1)

    # Вычисление эталона через NumPy
    C_numpy = np.dot(A, B)

    # Сравнение с учетом погрешности плавающей точки
    # rtol=1e-5, atol=1e-8 - стандартные допуски
    if np.allclose(C_cpp, C_numpy, rtol=1e-5, atol=1e-8):
        print("ВЕРИФИКАЦИЯ: УСПЕХ")
        print(f"Размер матрицы: {n}x{n}")
        print(f"Максимальная разница: {np.max(np.abs(C_cpp - C_numpy))}")
        sys.exit(0)
    else:
        print("ВЕРИФИКАЦИЯ: ПРОВАЛ (Результаты не совпадают)")
        print(f"Максимальная разница: {np.max(np.abs(C_cpp - C_numpy))}")
        sys.exit(1)


if __name__ == "__main__":
    main()