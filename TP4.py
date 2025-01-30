import numpy as np

total_nodes = 9

adjacency_matrix = np.full((total_nodes, total_nodes), float('inf'))

# Define edges (from node A to B with weight W)
graph_edges = [
    (1, 2, 4), (1, 5, 1), (1, 7, 2),
    (2, 3, 7), (2, 6, 5),
    (3, 4, 1), (3, 6, 8),
    (4, 6, 6), (4, 7, 4), (4, 8, 3),
    (5, 6, 9), (5, 7, 10),
    (6, 9, 2),
    (7, 9, 8),
    (8, 9, 1),
    (9, 8, 7)
]

# Populate the adjacency matrix with given edges
for start, end, weight in graph_edges:
    adjacency_matrix[start - 1][end - 1] = weight

def format_value(value):
    return ' inf' if value == float('inf') else f'{int(value):4d}'

formatted_matrix = np.array2string(
    adjacency_matrix,
    formatter={'all': format_value},
    separator=' ',
    max_line_width=120
)

print("Adjacency Matrix for Undirected and Weighted Graph:")
print(formatted_matrix)

# Function to implement Prim's Algorithm
def prim_mst(adjacency_matrix, total_nodes, root):
    selected_nodes = [False] * total_nodes
    selected_nodes[root] = True
    mst_edges = []
    total_weight = 0

    while len(mst_edges) < total_nodes - 1:
        min_edge = (None, None, float('inf'))
        for u in range(total_nodes):
            if selected_nodes[u]:
                for v in range(total_nodes):
                    if not selected_nodes[v] and adjacency_matrix[u][v] != float('inf'):
                        if adjacency_matrix[u][v] < min_edge[2]:
                            min_edge = (u, v, adjacency_matrix[u][v])

        u, v, weight = min_edge
        mst_edges.append((u + 1, v + 1, weight))
        total_weight += weight
        selected_nodes[v] = True

    return mst_edges, total_weight

# Disjoint Set for Kruskal's Algorithm
class DisjointSet:
    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, node):
        if self.parent[node] != node:
            self.parent[node] = self.find(self.parent[node])
        return self.parent[node]

    def union(self, node1, node2):
        root1 = self.find(node1)
        root2 = self.find(node2)
        if root1 != root2:
            if self.rank[root1] > self.rank[root2]:
                self.parent[root2] = root1
            elif self.rank[root1] < self.rank[root2]:
                self.parent[root1] = root2
            else:
                self.parent[root2] = root1
                self.rank[root1] += 1

# Function to implement Kruskal's Algorithm
def kruskal_mst(adjacency_matrix, total_nodes):
    edges_list = []
    for i in range(total_nodes):
        for j in range(i + 1, total_nodes):
            if adjacency_matrix[i][j] != float('inf'):
                edges_list.append((i, j, adjacency_matrix[i][j]))

    edges_list.sort(key=lambda edge: edge[2])
    disjoint_set = DisjointSet(total_nodes)
    mst_edges = []
    total_weight = 0

    for u, v, weight in edges_list:
        if disjoint_set.find(u) != disjoint_set.find(v):
            disjoint_set.union(u, v)
            mst_edges.append((u + 1, v + 1, weight))
            total_weight += weight

    return mst_edges, total_weight

# Get user input
root_index = int(input("\nEnter the root node for Prim's algorithm: ")) - 1

# Run Prim's and Kruskal's algorithms
prim_result_edges, prim_total_weight = prim_mst(adjacency_matrix, total_nodes, root_index)
kruskal_result_edges, kruskal_total_weight = kruskal_mst(adjacency_matrix, total_nodes)

# Display results
print("\nPrim's Algorithm MST:")
for edge in prim_result_edges:
    print(f"Edge: {edge[0]} - {edge[1]}, Weight: {edge[2]}")
print(f"Total weight of MST: {prim_total_weight}")

print("\nKruskal's Algorithm MST:")
for edge in kruskal_result_edges:
    print(f"Edge: {edge[0]} - {edge[1]}, Weight: {edge[2]}")
print(f"Total weight of MST: {kruskal_total_weight}")