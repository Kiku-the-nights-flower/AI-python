import random
from math import floor

type Graph = dict[int, set[int]]

def get_graph_conflicts(graph: Graph, colors: dict) -> dict:
    conflicts = {}
    for u in graph:
        conflicts[u] = set()
        for v in graph[u]:
            if colors[u] == colors[v]:
                conflicts[u].add(v)
    return conflicts

# return count of conflicts if the node would be of said color
def get_would_be_conflicts(graph: Graph, colors: dict, node: int, temp_color: int) -> list:
    return [neighbor for neighbor in graph[node] if colors[neighbor] == temp_color]

# call only after updating a color of a node, recalculates the conflicts of updated node + fixes conflicts in the neighbor nodes
def update_conflicts(graph: Graph, colors: dict, conflicts: Graph, updated_node: int):
    conflicts[updated_node] = set()  # reset conflicts for this node
    for neighbor in graph[updated_node]:
        if colors[neighbor] == colors[updated_node]:
            conflicts[updated_node].add(neighbor)
            conflicts[neighbor].add(updated_node)
        else:
            conflicts[neighbor].discard(updated_node)

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

# returns count of key/value pairs that have their value list empty
def count_non_empty(conflicts: dict[int, list]) -> int:
    return len([key for key, value in conflicts.items() if value])

def repaint(coloring: dict, conflicts: dict, color_count: int):
    vertex_count = len(coloring)
    recolor_count = floor((vertex_count / 100) * 10)  # recolor 5% of the vertices, found that more is not necessarily better
    recolored = set()
    for _ in range(recolor_count):
        if count_non_empty(conflicts) == 0:
            break
        random_node = random.choice(list(coloring.keys()))
        while random_node in recolored:
            random_node = random.choice(list(coloring.keys()))
        coloring[random_node] = random.randint(0, color_count - 1)
        recolored.add(random_node)

def color(graph: Graph, color_count: int, steps: int) -> (dict, bool):
    # init region
    colors = {u: random.randint(0, color_count - 1) for u in graph} # random coloring
    conflicts = get_graph_conflicts(graph, colors)
    last_conflicted = list(conflicts.keys())
    repaint_counter = 0
    doomsday_clock = 0

    for step in range(steps):
        if count_non_empty(conflicts) == 1:
            print("")

        if count_non_empty(conflicts) == 0:
            return colors, True

        # check the status each 200 iterations, and if stuck do an escape random repaint
        if step % 200 == 0:
            current_conflicted = [key for key, value in conflicts.items() if value]
            if last_conflicted == current_conflicted:
                doomsday_clock += 1
            else:
                doomsday_clock = 0
            if doomsday_clock >= 5:
                repaint_counter += 1
                repaint(colors, conflicts, color_count)
                conflicts = get_graph_conflicts(graph, colors)
                doomsday_clock = 0
            last_conflicted = current_conflicted

        # verbose debug
        if step % 50000 == 0:
            print(f"Step {step}: {count_non_empty(conflicts)} conflicts")

        # try to find a better color for a random node that has at least one conflict
        random_node = random.choice(list(conflicts.keys()))
        best_color = colors[random_node]
        current_conflict_count = len(conflicts[random_node])
        for c in range(color_count):
            new_conflicts = len(get_would_be_conflicts(graph, colors, random_node, c))
            if new_conflicts < current_conflict_count:
                current_conflict_count = new_conflicts
                best_color = c
            if new_conflicts == 0: # speedup if we find a "perfect" color
                break
        colors[random_node] = best_color
        update_conflicts(graph, colors, conflicts, random_node)
    return colors, False

def dynamic_color(graph: Graph, initial_color_count: int):
    current_color_count = initial_color_count
    solution = None
    steps_to_run = 50000000
    while current_color_count > 0:
        colors, success = color(graph, current_color_count, steps_to_run)
        if success:
            print(f"Success with {current_color_count} colors.")
            solution = colors
            current_color_count -= 1
        else:
            print(f"Failed to color with {current_color_count} colors")
            break
    return solution, current_color_count + 1

if __name__ == '__main__':
    graph = read_file("inputs/in_small.txt")
    initial_color_count = len(list(graph.keys()))
    solution, chromatic_number = dynamic_color(graph, initial_color_count)
    if solution:
        print(f"Found valid coloring with {chromatic_number} colors.")
        print(solution)
    else:
        print("No valid coloring found within the given steps.")
