#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <string>
#include <windows.h> // Для поддержки русского языка в консоли

using namespace std;

// Структура для хранения метаданных
struct TaskMetadata {
    int size;
    double time_ms;
    long long flops;
    long long memory_bytes; // Объем памяти в байтах
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
    file << "MEMORY_BYTES: " << meta.memory_bytes << endl; // Вывод объема памяти
    file << "# METADATA_END" << endl;

    file.close();
    return true;
}

// Функция умножения матриц
void multiplyMatrices(const vector<vector<double>>& A, const vector<vector<double>>& B, vector<vector<double>>& C, int n) {
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

int main(int argc, char* argv[]) {
    SetConsoleOutputCP(65001); // Поддержка русского языка в Windows

    if (argc != 4) {
        cerr << "Использование: " << argv[0] << " <файл_матрицы_A> <файл_матрицы_B> <файл_результата>" << endl;
        return 1;
    }

    string fileA = argv[1];
    string fileB = argv[2];
    string fileRes = argv[3];

    vector<vector<double>> A, B, C;
    int nA, nB;

    cout << "Чтение исходных данных..." << endl;
    if (!readMatrix(fileA, A, nA)) return 1;
    if (!readMatrix(fileB, B, nB)) return 1;

    if (nA != nB) {
        cerr << "Ошибка: Матрицы должны быть квадратными и одинакового размера!" << endl;
        return 1;
    }
    int n = nA;

    cout << "Размер матрицы: " << n << "x" << n << endl;
    cout << "Выполнение умножения..." << endl;

    // Замер времени
    auto start = chrono::high_resolution_clock::now();
    multiplyMatrices(A, B, C, n);
    auto end = chrono::high_resolution_clock::now();

    chrono::duration<double, milli> duration = end - start;
    double time_ms = duration.count();

    // Оценка объема вычислений (2 * N^3 операций)
    long long flops = 2LL * n * n * n;

    // Расчет объема памяти (3 матрицы * N*N * 8 байт)
    long long memory_bytes = 3LL * n * n * sizeof(double);

    TaskMetadata meta = {n, time_ms, flops, memory_bytes};

    if (!writeResult(fileRes, C, meta)) return 1;

    cout << "Результат записан в " << fileRes << endl;
    cout << "Время выполнения: " << time_ms << " мс" << endl;
    cout << "Объем задачи (FLOPs): " << flops << endl;
    cout << "Объем памяти: " << memory_bytes << " байт" << endl;

    return 0;
}