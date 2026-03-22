#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <string>
#include <cmath>
#include <omp.h>  // Подключение OpenMP

using namespace std;

// Структура для хранения метаданных
struct TaskMetadata {
    int size;
    double time_ms;
    long long flops;
    int threads_used;
};

// Функция чтения матрицы из файла
bool readMatrix(const string& filename, vector<vector<double>>& matrix, int& n) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Ошибка: Не удалось открыть файл " << filename << endl;
        return false;
    }

    if (!(file >> n)) {
        cerr << "Ошибка: Не удалось прочитать размер матрицы из " << filename << endl;
        return false;
    }

    matrix.resize(n, vector<double>(n));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (!(file >> matrix[i][j])) {
                cerr << "Ошибка: Не удалось прочитать элемент матрицы [" << i << "][" << j << "]" << endl;
                return false;
            }
        }
    }
    file.close();
    return true;
}

// Функция записи матрицы и метаданных в файл
bool writeResult(const string& filename, const vector<vector<double>>& matrix, const TaskMetadata& meta) {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "Ошибка: Не удалось создать файл результата " << filename << endl;
        return false;
    }

    int n = matrix.size();
    file << n << endl;
    file << fixed << setprecision(10);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            file << matrix[i][j] << " ";
        }
        file << endl;
    }

    // Блок метаданных
    file << "# METADATA_START" << endl;
    file << "SIZE: " << meta.size << endl;
    file << "TIME_MS: " << meta.time_ms << endl;
    file << "FLOPS: " << meta.flops << endl;
    file << "THREADS: " << meta.threads_used << endl;
    file << "# METADATA_END" << endl;

    file.close();
    return true;
}

// Параллельная функция умножения матриц с OpenMP
void multiplyMatricesOMP(const vector<vector<double>>& A, const vector<vector<double>>& B,
                         vector<vector<double>>& C, int n, int num_threads) {
    C.resize(n, vector<double>(n, 0.0));

    // Установка количества потоков для этой области
    omp_set_num_threads(num_threads);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            for (int k = 0; k < n; ++k) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
}

// Последовательная версия для сравнения (baseline)
void multiplyMatricesSeq(const vector<vector<double>>& A, const vector<vector<double>>& B,
                         vector<vector<double>>& C, int n) {
    C.resize(n, vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            for (int k = 0; k < n; ++k) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
}

void printUsage(const char* progName) {
    cout << "Использование:" << endl;
    cout << "  " << progName << " <файл_А> <файл_В> <файл_результата> [threads]" << endl;
    cout << "  threads - количество потоков (по умолчанию: все доступные)" << endl;
    cout << "  Для последовательного режима: threads=1" << endl;
}

int main(int argc, char* argv[]) {
    if (argc < 4 || argc > 5) {
        printUsage(argv[0]);
        return 1;
    }

    string fileA = argv[1];
    string fileB = argv[2];
    string fileRes = argv[3];
    int num_threads = (argc == 5) ? stoi(argv[4]) : omp_get_max_threads();

    // Ограничиваем количество потоков доступными ядрами
    int max_threads = omp_get_num_procs();
    if (num_threads > max_threads) {
        cout << "Предупреждение: запрошено " << num_threads << " потоков, но доступно только "
             << max_threads << " ядер. Используем " << max_threads << "." << endl;
        num_threads = max_threads;
    }

    vector<vector<double>> A, B, C;
    int nA, nB;

    cout << "=== Параллельное умножение матриц (OpenMP) ===" << endl;
    cout << "Чтение исходных данных..." << endl;

    if (!readMatrix(fileA, A, nA)) return 1;
    if (!readMatrix(fileB, B, nB)) return 1;

    if (nA != nB) {
        cerr << "Ошибка: Матрицы должны быть квадратными и одинакового размера!" << endl;
        return 1;
    }
    int n = nA;

    cout << "Размер матрицы: " << n << "x" << n << endl;
    cout << "Количество потоков: " << num_threads << " (доступно ядер: " << omp_get_num_procs() << ")" << endl;
    cout << "Выполнение умножения..." << endl;

    // Замер времени
    auto start = chrono::high_resolution_clock::now();
    multiplyMatricesOMP(A, B, C, n, num_threads);
    auto end = chrono::high_resolution_clock::now();

    chrono::duration<double, milli> duration = end - start;
    double time_ms = duration.count();

    // Оценка объема задачи (2 * N^3 операций)
    long long flops = 2LL * n * n * n;
    double gflops = (flops / 1e9) / (time_ms / 1000.0);

    TaskMetadata meta = { n, time_ms, flops, num_threads };

    if (!writeResult(fileRes, C, meta)) return 1;

    cout << "\n=== Результаты ===" << endl;
    cout << "Результат записан в " << fileRes << endl;
    cout << "Время выполнения: " << time_ms << " мс" << endl;
    cout << "Объем задачи: " << flops << " FLOPs" << endl;
    cout << "Производительность: " << fixed << setprecision(3) << gflops << " GFLOPS" << endl;

    return 0;
}