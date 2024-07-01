//
// Created by pedda on 8-4-24.
//

#ifndef L_INFINITY_TESTING_GRAPH_H
#define L_INFINITY_TESTING_GRAPH_H


enum class Norm {
    INF, ONE
};


#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseQR>

typedef double double_custom;
typedef Eigen::Matrix<double_custom, Eigen::Dynamic, Eigen::Dynamic> MatrixXq;
//typedef Eigen::MatrixXd MatrixXq;
typedef Eigen::Matrix<double_custom, Eigen::Dynamic, 1> VectorXq;
//typedef Eigen::VectorXd VectorXq;

class Graph {
public:
    int n;
    int m;
    std::vector<std::vector<std::pair<int,int>>> adj_list;
    std::vector<std::pair<int,int>> edges;
    std::vector<std::vector<bool>> adj_matrix;
    VectorXq demand;
    MatrixXq matrix;
    VectorXq sparse_x;

    Graph(int n, int m);

    void randomize();
    VectorXq get_demand();
    MatrixXq get_matrix();

private:
    int current = 0;
    void build_tree();
    void build_rest();
    void generate_matrix();
    void generate_demand();
    void add_edge(int u, int v);
    bool dfs(MatrixXq& D, int current, int origin, int target, int current_column);
    void add_edges(int num);
};


#endif //L_INFINITY_TESTING_GRAPH_H
