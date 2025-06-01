#include <vector>
#include <random>
#include <chrono>
#include <iostream>

std::vector<float> generate_random_matrix(int rows, int cols) {
    std::vector<float> mat(rows * cols);

    std::mt19937 mt_gen{static_cast<std::mt19937::result_type>(std::chrono::steady_clock::now().time_since_epoch().count())};

    std::uniform_real_distribution<float> matvalue_dist{-10.0, 10.0};

    for (int i = 0; i < mat.size(); ++i) {
        mat[i] = matvalue_dist(mt_gen);
    }

    return mat;
}
