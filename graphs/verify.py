
type Graph = dict[int, set[int]]

def get_graph_conflicts(graph: Graph, colors: dict) -> dict:
    conflicts = {}
    for u in graph:
        conflicts[u] = []
        for v in graph[u]:
            if colors[u] == colors[v]:
                conflicts[u].append(v)
    return conflicts

def read_file(filepath: str) -> Graph:
    graph = {}
    with open(filepath) as f:
        for line in f:
            if line.startswith('p'):
                parts = line.split()
                edge_count = int(parts[2])
                for i in range(edge_count):
                    graph[i] = set()
            elif line.startswith('e'):
                parts = list(map(int, line.split()[1:]))
                u, v = parts[0]-1, parts[1]-1
                graph[u].add(v)
                graph[v].add(u)
    return graph

def count_non_empty(conflicts: dict[int, list]) -> int:
    return len([key for key, value in conflicts.items() if value])


if __name__ == "__main__":
    graph = read_file("inputs/in.txt")
    colors = {0: 1, 1: 19, 2: 36, 3: 9, 4: 35, 5: 21, 6: 26, 7: 6, 8: 3, 9: 28, 10: 43, 11: 5, 12: 17, 13: 11, 14: 15, 15: 31, 16: 39, 17: 24, 18: 34, 19: 33, 20: 34, 21: 27, 22: 29, 23: 8, 24: 3, 25: 13, 26: 1, 27: 14, 28: 19, 29: 38, 30: 0, 31: 46, 32: 35, 33: 12, 34: 11, 35: 33, 36: 7, 37: 5, 38: 9, 39: 19, 40: 27, 41: 5, 42: 25, 43: 24, 44: 32, 45: 39, 46: 26, 47: 45, 48: 29, 49: 26, 50: 18, 51: 11, 52: 6, 53: 46, 54: 3, 55: 32, 56: 44, 57: 22, 58: 21, 59: 17, 60: 42, 61: 41, 62: 35, 63: 37, 64: 10, 65: 23, 66: 14, 67: 1, 68: 7, 69: 4, 70: 44, 71: 2, 72: 31, 73: 7, 74: 13, 75: 30, 76: 44, 77: 20, 78: 41, 79: 43, 80: 45, 81: 42, 82: 15, 83: 2, 84: 16, 85: 18, 86: 0, 87: 30, 88: 45, 89: 25, 90: 12, 91: 4, 92: 8, 93: 27, 94: 16, 95: 34, 96: 6, 97: 4, 98: 43, 99: 23, 100: 38, 101: 42, 102: 35, 103: 6, 104: 28, 105: 31, 106: 2, 107: 40, 108: 37, 109: 29, 110: 20, 111: 0, 112: 10, 113: 10, 114: 2, 115: 17, 116: 16, 117: 24, 118: 33, 119: 39, 120: 21, 121: 40, 122: 22, 123: 28, 124: 14}
    print(count_non_empty(get_graph_conflicts(graph, colors)))
