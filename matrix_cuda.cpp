%%writefile matrix_cuda.cu
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

using namespace std;

// CUDA KERNEL
__global__ void matrixMulKernel(const double* A, const double* B, double* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        double sum = 0.0;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

vector<vector<double>> readMatrix(const string& filename, int& n) {
    ifstream file(filename);
    file >> n;
    vector<vector<double>> matrix(n, vector<double>(n));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            file >> matrix[i][j];
    return matrix;
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        cout << "Usage: " << argv[0] << " <matrix1> <matrix2> <block_size>" << endl;
        return 1;
    }
    
    string fileA = argv[1];
    string fileB = argv[2];
    int blockSize = atoi(argv[3]);
    
    int n;
    auto A = readMatrix(fileA, n);
    auto B = readMatrix(fileB, n);
    
    cout << "Matrix size: " << n << "x" << n << endl;
    cout << "Block size: " << blockSize << "x" << blockSize << endl;
    
    // Конвертация в плоские массивы
    vector<double> h_A(n * n), h_B(n * n), h_C(n * n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            h_A[i * n + j] = A[i][j];
            h_B[i * n + j] = B[i][j];
        }
    }
    
    // Выделение памяти на GPU
    double *d_A, *d_B, *d_C;
    size_t bytes = n * n * sizeof(double);
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);
    
    // Копирование на GPU
    cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice);
    
    // Настройка сетки
    dim3 threads(blockSize, blockSize);
    dim3 blocks((n + blockSize - 1) / blockSize, (n + blockSize - 1) / blockSize);
    
    // Прогрев
    matrixMulKernel<<<blocks, threads>>>(d_A, d_B, d_C, n);
    cudaDeviceSynchronize();
    
    // Замер времени
    auto start = chrono::high_resolution_clock::now();
    
    matrixMulKernel<<<blocks, threads>>>(d_A, d_B, d_C, n);
    cudaDeviceSynchronize();
    
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> duration = end - start;
    
    // Копирование результата обратно
    cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost);
    
    // Проверка (сумма элементов)
    double sum = 0;
    for (int i = 0; i < n * n; i++) sum += h_C[i];
    cout << "Result sum: " << sum << endl;
    
    // Вывод
    long long ops = 2LL * n * n * n;
    double gflops = ops / (duration.count() * 1e9);
    
    cout << "Execution time: " << duration.count() << " seconds" << endl;
    cout << "Performance: " << gflops << " GFLOPS" << endl;
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    return 0;
}