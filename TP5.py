import numpy as np
import heapq

vertices = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'L', 'M']
num_vertices = len(vertices)
vertex_to_index = {vertex: idx for idx, vertex in enumerate(vertices)}

adjacency_matrix = [[float('inf')] * num_vertices for _ in range(num_vertices)]

edges = [
    ('A', 'C', 1), ('A', 'B', 4),
    ('B', 'F', 3),
    ('C', 'D', 8), ('C', 'F', 7),
    ('D', 'H', 5),
    ('F', 'H', 1), ('F', 'E', 1),
    ('E', 'H', 2),
    ('H', 'G', 3), ('H', 'M', 7), ('H', 'L', 6),
    ('G', 'M', 4),
    ('M', 'L', 1),
    ('L', 'G', 4), ('L', 'E', 2)
]

target_index = adjacency_matrix
target_dict = vertex_to_index
for start, end, weight in edges:
    i, j = target_dict[start], target_dict[end]
    target_index[i][j] = weight
    target_index[j][i] = weight

adjacency_matrix_np = np.array(target_index)

def format_output(value):
    return ' inf' if value == float('inf') else f'{int(value):4d}'

formatted_matrix = np.array2string(
    adjacency_matrix_np,
    formatter={'all': format_output},
    separator=' ',
    max_line_width=120
)

print("Adjacency Matrix for Undirected & Weighted graph:")
print(formatted_matrix)

index_to_vertex = {idx: vertex for idx, vertex in enumerate(vertices)}

def dijkstra(adjacency_matrix, start_vertex, end_vertex):
    start_idx = vertex_to_index[start_vertex]
    end_idx = vertex_to_index[end_vertex]

    distances = [float('inf')] * num_vertices
    distances[start_idx] = 0
    previous_nodes = [None] * num_vertices
    priority_queue = [(0, start_idx)]

    while priority_queue:
        current_distance, current_vertex = heapq.heappop(priority_queue)

        if current_distance > distances[current_vertex]:
            continue

        for neighbor in range(num_vertices):
            edge_weight = adjacency_matrix[current_vertex][neighbor]
            if edge_weight != float('inf'):
                new_distance = current_distance + edge_weight
                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    previous_nodes[neighbor] = current_vertex
                    heapq.heappush(priority_queue, (new_distance, neighbor))

    path = []
    current = end_idx
    while current is not None:
        path.append(index_to_vertex[current])
        current = previous_nodes[current]
    path.reverse()

    return path, distances[end_idx]

start_vertex = input("Enter the source node (A-M): ").strip().upper()
end_vertex = input("Enter the target node (A-M): ").strip().upper()

if start_vertex in vertex_to_index and end_vertex in vertex_to_index:
    shortest_path, path_weight = dijkstra(adjacency_matrix, start_vertex, end_vertex)
    print(f"Shortest path from {start_vertex} to {end_vertex}: {' -> '.join(shortest_path)}")
    print(f"Weighted sum of the shortest path: {path_weight}")
else:
    print("Invalid nodes. Enter valid node labels (A-M)")
