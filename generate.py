import sys
import random


def main():
    if len(sys.argv) != 4:
        print("Usage: python generate.py <size> <file_a> <file_b>")
        sys.exit(1)

    size = int(sys.argv[1])
    file_a = sys.argv[2]
    file_b = sys.argv[3]

    print(f"Generating matrix A ({size}x{size})...")
    with open(file_a, 'w') as f:
        f.write(f"{size}\n")
        for i in range(size):
            row = [f"{random.uniform(-100, 100):.6f}" for _ in range(size)]
            f.write(" ".join(row) + "\n")

    print(f"Generating matrix B ({size}x{size})...")
    with open(file_b, 'w') as f:
        f.write(f"{size}\n")
        for i in range(size):
            row = [f"{random.uniform(-100, 100):.6f}" for _ in range(size)]
            f.write(" ".join(row) + "\n")

    print(f"✓ Matrices saved to {file_a} and {file_b}")


if __name__ == "__main__":
    main()
