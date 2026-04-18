#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <string>
#include <cmath>
#include <mpi.h>

using namespace std;

// Структура для хранения метаданных
struct TaskMetadata {
    int size;
    double time_ms;
    long long flops;
    int processes;
    int rank;
};

// Функция чтения матрицы из файла (только для процесса 0)
bool readMatrix(const string& filename, vector<double>& matrix, int& n) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Ошибка: Не удалось открыть файл " << filename << endl;
        return false;
    }

    if (!(file >> n)) {
        cerr << "Ошибка: Не удалось прочитать размер матрицы из " << filename << endl;
        return false;
    }

    matrix.resize(n * n);
    for (int i = 0; i < n * n; ++i) {
        if (!(file >> matrix[i])) {
            cerr << "Ошибка: Не удалось прочитать элемент матрицы [" << i << "]" << endl;
            return false;
        }
    }
    file.close();
    return true;
}

// Функция записи результата (только для процесса 0)
bool writeResult(const string& filename, const vector<double>& matrix, const TaskMetadata& meta) {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "Ошибка: Не удалось создать файл результата " << filename << endl;
        return false;
    }

    int n = meta.size;
    file << n << endl;
    file << fixed << setprecision(10);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            file << matrix[i * n + j] << " ";
        }
        file << endl;
    }

    // Блок метаданных
    file << "# METADATA_START" << endl;
    file << "SIZE: " << meta.size << endl;
    file << "TIME_MS: " << meta.time_ms << endl;
    file << "FLOPS: " << meta.flops << endl;
    file << "PROCESSES: " << meta.processes << endl;
    file << "# METADATA_END" << endl;

    file.close();
    return true;
}

// Параллельное умножение матриц с использованием MPI
void multiplyMatricesMPI(const vector<double>& A, const vector<double>& B,
                         vector<double>& C, int n, int rank, int size) {

    // Рассчитываем количество строк для каждого процесса
    int rows_per_proc = n / size;
    int remainder = n % size;

    // Для процесса rank: определяем диапазон строк
    int start_row = rank * rows_per_proc + min(rank, remainder);
    int end_row = start_row + rows_per_proc + (rank < remainder ? 1 : 0);
    int local_rows = end_row - start_row;

    // Локальные матрицы
    vector<double> local_A(local_rows * n);
    vector<double> local_C(local_rows * n, 0.0);

    // Копируем свою часть матрицы A
    for (int i = 0; i < local_rows; ++i) {
        for (int j = 0; j < n; ++j) {
            local_A[i * n + j] = A[(start_row + i) * n + j];
        }
    }

    // Матрица B одинакова для всех процессов
    vector<double> local_B = B;

    // Локальное умножение
    for (int i = 0; i < local_rows; ++i) {
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            for (int k = 0; k < n; ++k) {
                sum += local_A[i * n + k] * local_B[k * n + j];
            }
            local_C[i * n + j] = sum;
        }
    }

    // Собираем результаты на процессе 0
    // Сначала отправляем размеры для каждого процесса
    vector<int> recv_counts(size);
    vector<int> displs(size);

    for (int i = 0; i < size; ++i) {
        int proc_rows = rows_per_proc + (i < remainder ? 1 : 0);
        recv_counts[i] = proc_rows * n;
        displs[i] = (i == 0) ? 0 : displs[i-1] + recv_counts[i-1];
    }

    MPI_Gatherv(local_C.data(), local_rows * n, MPI_DOUBLE,
                C.data(), recv_counts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 4) {
        if (rank == 0) {
            cerr << "Использование: " << argv[0]
                 << " <файл_матрицы_A> <файл_матрицы_B> <файл_результата>" << endl;
        }
        MPI_Finalize();
        return 1;
    }

    string fileA = argv[1];
    string fileB = argv[2];
    string fileRes = argv[3];

    vector<double> A, B, C;
    int n = 0;

    // Процесс 0 читает матрицы
    if (rank == 0) {
        cout << "=== Параллельное умножение матриц (MPI) ===" << endl;
        cout << "Количество процессов: " << size << endl;
        cout << "Чтение исходных данных..." << endl;

        if (!readMatrix(fileA, A, n)) {
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        if (!readMatrix(fileB, B, n)) {
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        cout << "Размер матрицы: " << n << "x" << n << endl;
        cout << "Выполнение умножения..." << endl;
    }

    // Рассылаем размер матрицы всем процессам
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Рассылаем матрицы всем процессам
    if (rank != 0) {
        A.resize(n * n);
        B.resize(n * n);
    }
    MPI_Bcast(A.data(), n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(B.data(), n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Инициализируем матрицу результата
    if (rank == 0) {
        C.resize(n * n);
    }

    // Синхронизация перед замером времени
    MPI_Barrier(MPI_COMM_WORLD);

    // Замер времени
    double start_time = MPI_Wtime();
    multiplyMatricesMPI(A, B, C, n, rank, size);
    double end_time = MPI_Wtime();

    double time_sec = end_time - start_time;
    double time_ms = time_sec * 1000.0;

    // Оценка объема задачи
    long long flops = 2LL * n * n * n;

    if (rank == 0) {
        TaskMetadata meta = { n, time_ms, flops, size, rank };

        if (!writeResult(fileRes, C, meta)) {
            MPI_Finalize();
            return 1;
        }

        double gflops = (flops / 1e9) / time_sec;

        cout << "\n=== Результаты ===" << endl;
        cout << "Результат записан в " << fileRes << endl;
        cout << "Время выполнения: " << time_ms << " мс" << endl;
        cout << "Объем задачи: " << flops << " FLOPs" << endl;
        cout << "Производительность: " << fixed << setprecision(3)
             << gflops << " GFLOPS" << endl;
    }

    MPI_Finalize();
    return 0;
}