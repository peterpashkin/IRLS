//
// Created by pedda on 8-4-24.
//

#include <iostream>
#include "Graph.h"

Graph::Graph(int n, int m) : n(n), m(m), adj_list(n), matrix(n, m), demand(m), adj_matrix(n, std::vector<bool>(n, false)) {
    //randomize();
}


void Graph::randomize() {
    build_tree(); // generates a random tree over the graph
    build_rest(); // generates all the other edges, multi-edges are allowed
    generate_demand(); // generates a random demand vector by computing A*x = b and choosing x sparse {1,0,-1} vector
    generate_matrix(); // generates the matrix A
}

VectorXq Graph::get_demand() {
    return demand;
}

MatrixXq Graph::get_matrix() {
    return matrix;
}

void Graph::build_tree() {
    std::vector<int> pruefer(n-2);
    for(int i=0; i<n-2; i++) {
        pruefer[i] = rand() % n;
    }
    std::vector<int> degree(n, 0);
    for(int i=0; i<n-2; i++) {
        degree[pruefer[i]]++;
    }

    for(int i=0; i<n-2; i++) {
        // determine smallest element not in degree or workset
        int elem = 0;
        while(degree[elem]) ++elem;
        int v = pruefer[i];
        add_edge(v, elem);
        degree[v]--;
        degree[elem]++;
    }
    // find two numbers not in degree
    int u = 0;
    while(degree[u]) ++u;
    ++degree[u];
    int v = 0;
    while(degree[v]) ++v;
    add_edge(u,v);
}

void Graph::add_edges(int num) {
    for (int i = 0; i <= num; i++) {
        int u = rand() % n;
        int v = rand() % n;
        if(u == v || adj_matrix[u][v]) {--i; continue;}
        add_edge(u, v);
    }
}

void Graph::build_rest() {
    add_edges(m-n);
}

void Graph::add_edge(int u, int v) {
    adj_list[u].emplace_back(v, current);
    adj_list[v].emplace_back(u, current);
    edges.emplace_back(u, v);
    adj_matrix[u][v] = true;
    adj_matrix[v][u] = true;
    ++current;
}

void Graph::generate_matrix() {
    int column = 0;
    for(auto edge : edges) {
        matrix(edge.first, column) = 1;
        matrix(edge.second, column) = -1;
        ++column;
    }
}

void Graph::generate_demand() {
    // create a random sparse vector with 16 +1/-1 entries, there should be 8 +1 and 8 -1
    VectorXq x = VectorXq::Zero(m);

    if(m < 32) {
        //set half the edges to 1
        for (int i = 0; i < m/2; i++) {
            x(i) = 1;
        }
    } else {

        for (int i = 0; i < 8; i++) {
            int index = rand() % m;
            while (x(index) != 0) {
                index = rand() % m;
            }
            x(index) = 1;
        }
        for (int i = 0; i < 8; i++) {
            int index = rand() % m;
            while (x(index) != 0) {
                index = rand() % m;
            }
            x(index) = -1;
        }
    }

    sparse_x = x;

    VectorXq b = VectorXq::Zero(n);
    for(int i=0; i<n; i++) {
        for(auto edge : adj_list[i]) {
            b(i) += x(edge.second) * (edges[edge.second].first == i ? 1 : -1);
        }
    }
    demand = b;
}

bool Graph::dfs(MatrixXq& D, int current, int origin, int target, int current_column) {
    if(current == target) {
        return true;
    }
    for(auto edge : adj_list[current]) {
        if(edge.second >= n-1) break; // break should be ok because after that only pairs with second >= n-1 should be in the list
        if(edge.first == origin) continue;
        if(dfs(D, edge.first, current, target, current_column)) {
            D(edge.second, current_column) = matrix(current, edge.second) == 1 ? 1 : -1;
            return true;
        }
    }
    return false;
}


