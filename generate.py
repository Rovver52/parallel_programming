import sys
import numpy as np


def generate_matrix(filename, n):
    matrix = np.random.rand(n, n)
    with open(filename, 'w') as f:
        f.write(f"{n}\n")
        for row in matrix:
            f.write(" ".join(map(str, row)) + "\n")
    print(f"Сгенерирована матрица {n}x{n} в файле {filename}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Использование: python generate.py <размер_N> <файл_A> <файл_B>")
        sys.exit(1)

    n = int(sys.argv[1])
    generate_matrix(sys.argv[2], n)
    generate_matrix(sys.argv[3], n)