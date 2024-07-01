#pragma once
#include "Runner.h"
#include "graph_generation/Graph.h"
#include <chrono>
#include <iostream>
#include <fstream>
#define GRAPHS 20


Runner r{};

void assert_correct_infeasible_l1(const MatrixXq& A, const VectorXq& c, const VectorXq& b, double_custom M, double_custom eps) {
    double_custom val1 = b.transpose() * c;
    double_custom val2 = (A.transpose() * c).lpNorm<Eigen::Infinity>();
    assert(val1 / val2 >= (1-eps) * M);
}

void assert_correct_feasible_l1(const VectorXq x, double_custom M, double_custom eps) {
    double_custom val1 = x.lpNorm<1>();
    double_custom val2 = (1+eps) * M;
    assert(val1 <= val2);
}


void assert_correct_infeasible_linf(const MatrixXq& A, const VectorXq& c, const VectorXq& b, double_custom M, double_custom eps) {
    double_custom value = b.transpose() * (A * c.asDiagonal().inverse() * A.transpose()).completeOrthogonalDecomposition().pseudoInverse() * b;
    double_custom comparison = (1-eps) * (1-eps) * M * M;
    //std::cout << "Value: " << static_cast<double>(value) << " Comparison: " << static_cast<double>(comparison) << std::endl;
    assert(value + 1e-6 >= comparison);
    if(comparison > value) {
        std::cout << "Error of: " << static_cast<double>(comparison-value) << " with value of " << static_cast<double>(value) << std::endl;
    }
}

void assert_correct_feasible_linf(const VectorXq x, double_custom M, double_custom eps) {
    double_custom val1 = x.lpNorm<Eigen::Infinity>();
    double_custom val2 = (1+eps) * M;
    //std::cout << "Val1: " << static_cast<double>(val1) << " Val2: " << static_cast<double>(val2) << std::endl;
    assert(x.lpNorm<Eigen::Infinity>() <= (1+eps) * M + 1e-6);
}

VectorXq binary_search_over_M(const VectorXq& b, double_custom eps, Norm norm, Graph& g, bool recursive) {
    double_custom low = 0, high = 1;
    double_custom exp = 1;
    while(true) {
        VectorXq tmp;
        if(norm == Norm::INF) {
            r.linf_minimization(b, eps, exp, g);
        } else if(norm == Norm::ONE) {
            r.l1_minimization(b, eps, exp, g);
        }

        if(!r.wrong_approx) {
            break;
        }
        low = exp;
        exp *= 2;
        high = exp;
    }

    double_custom last_valid = low;
    const double_custom search_error = 1e-9Q;

    VectorXq x;
    while (low*(1.0+search_error) < high) {
        double_custom M = low + (high - low) / 2.0;
        if(norm == Norm::ONE) {
            x = r.l1_minimization(b, eps, M, g);
        } else if(norm == Norm::INF) {
            x = r.linf_minimization(b, eps, M, g);
        }

        if (!r.wrong_approx) {
            last_valid = M;
            high = M;
        } else {
            low = M;
            MatrixXq A = g.matrix;
            if(norm == Norm::INF) {
                assert_correct_infeasible_linf(A, x, b, M, eps);
            } else {
                assert_correct_infeasible_l1(A, x, b, M, eps);
            }
        }
    }

    if(norm == Norm::INF) {
        x = r.linf_minimization(b, eps, last_valid, g);
        assert_correct_feasible_linf(x, last_valid, eps);
    } else if(norm == Norm::ONE) {
        x = r.l1_minimization(b, eps, last_valid, g);
        assert_correct_feasible_l1(x, last_valid, eps);
    }

    return x;
}

int main() {
    int n = 150; // number of nodes
    int m = 200; // number of edges

    using std::vector;
    vector<Graph> graphs(GRAPHS, Graph(n, m));
    for(auto& g : graphs) {
        g.randomize();
    }

    vector<vector<Graph>> graphs_m(31);
    for(int i=1; i<=30; i++) {
        graphs_m[i] = vector<Graph>(GRAPHS, Graph(n, m * i));
        for(auto& g : graphs_m[i]) {
            g.randomize();
        }
    }


    double_custom eps = 0.5;

    for(int k=1; k<=12; k++) {
        for(int i=0; i < GRAPHS; i++) {
            Graph& g = graphs[i];
            VectorXq b = g.get_demand();
#ifdef INFINITY_FLAG
    #ifdef RECURSIVE_FLAG
            VectorXq x = binary_search_over_M(b,eps, Norm::INF, g, true);
    #else
            VectorXq x = binary_search_over_M(b,eps, Norm::INF, g, false);
    #endif
#else
    #ifdef RECURSIVE_FLAG
            VectorXq x = binary_search_over_M(b, eps, Norm::ONE, g, true);
    #else
            VectorXq x = binary_search_over_M(b, eps, Norm::ONE, g, false);
    #endif
#endif
        std::cout << "done" << std::endl;
        }
        eps *= 0.5;

    }


    //--------------- tests for increasing m ----------------
    std::cout << "Testing for increasing m" << std::endl;

    eps = 0.01;

    for(int k=1; k<=30; k++) {
        for(int i=0; i < GRAPHS; i++) {
            Graph& g = graphs_m[k][i];
            VectorXq b = g.get_demand();
#ifdef INFINITY_FLAG
    #ifdef RECURSIVE_FLAG
        VectorXq x = binary_search_over_M(b,eps, Norm::INF, g, true);
    #else
        VectorXq x = binary_search_over_M(b,eps, Norm::INF, g, false);
    #endif
#else
    #ifdef RECURSIVE_FLAG
        VectorXq x = binary_search_over_M(b, eps, Norm::ONE, g, true);
    #else
        VectorXq x = binary_search_over_M(b, eps, Norm::ONE, g, false);
    #endif
#endif
        }
    }




}