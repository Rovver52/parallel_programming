import sys
import numpy as np


def read_matrix_cpp(filename):
    """Чтение результата из файла C++ программы"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        n = int(lines[0].strip())
        matrix = []
        for i in range(1, n + 1):
            row = [float(x) for x in lines[i].split()]
            matrix.append(row)
        return np.array(matrix), n
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None, None


def read_input_matrix(filename):
    """Чтение входной матрицы"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read().split()
        iterator = iter(content)
        n = int(next(iterator))
        matrix = [[float(next(iterator)) for _ in range(n)] for _ in range(n)]
        return np.array(matrix)
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None


def main():
    if len(sys.argv) != 4:
        print("Usage: python verify_omp.py <matrix_A> <matrix_B> <result_file>")
        sys.exit(1)

    file_a, file_b, file_res = sys.argv[1], sys.argv[2], sys.argv[3]

    print("=== Verification (OpenMP result) ===")

    A = read_input_matrix(file_a)
    B = read_input_matrix(file_b)
    C_cpp, n = read_matrix_cpp(file_res)

    if A is None or B is None or C_cpp is None:
        print("VERIFICATION: FAILED (File read error)")
        sys.exit(1)

    # Эталонное умножение через NumPy
    C_numpy = np.dot(A, B)

    # Проверка с учётом погрешности floating-point
    max_diff = np.max(np.abs(C_cpp - C_numpy))
    if np.allclose(C_cpp, C_numpy, rtol=1e-5, atol=1e-8):
        print("✅ VERIFICATION: SUCCESS")
        print(f"   Matrix size: {n}x{n}")
        print(f"   Max difference: {max_diff:.2e}")
        sys.exit(0)
    else:
        print("❌ VERIFICATION: FAILED (Results mismatch)")
        print(f"   Max difference: {max_diff:.2e}")
        print(f"   Recommended tolerance: rtol=1e-5, atol=1e-8")
        sys.exit(1)


if __name__ == "__main__":
    main()