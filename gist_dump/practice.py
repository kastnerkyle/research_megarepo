# Author: Kyle Kastner

graph = {'A': set(['B', 'C']),
         'B': set(['A', 'D', 'E']),
         'C': set(['A', 'F']),
         'D': set(['B']),
         'E': set(['B', 'F']),
         'F': set(['C', 'E'])}


def dfs(graph, start):
    stack = [start]
    visited = set()
    while stack:
        vertex = stack.pop()
        if vertex not in visited:
            visited.add(vertex)
            stack.extend(graph[vertex] - visited)
    return visited


def bfs(graph, start):
    stack = [start]
    visited = set()
    while stack:
        vertex = stack.pop(0)
        if vertex not in visited:
            visited.add(vertex)
            stack.extend(graph[vertex] - visited)
    return visited


def dfs_paths(graph, start, end):
    stack = [(start, [start])]
    while stack:
        vertex, path = stack.pop()
        for nx in graph[vertex] - set(path):
            if nx == end:
                yield path + [end]
            else:
                stack.append((nx, path + [nx]))


def bfs_paths(graph, start, end):
    queue = [(start, [start])]
    while queue:
        vertex, path = queue.pop(0)
        for nx in graph[vertex] - set(path):
            if nx == end:
                yield path + [end]
            else:
                queue.append((nx, path + [nx]))

print(dfs(graph, 'A'))
print(bfs(graph, 'A'))
print([p for p in dfs_paths(graph, 'A', 'E')])
print([p for p in bfs_paths(graph, 'A', 'E')])


strings = ["cat", "dog", "pony"]
hash_list = []
for i in range(len(strings)):
    hash_list.append(hash(strings[i]) % 100)
print(hash_list)
print(hash("pony") % 100 in hash_list)
print(hash("ponies") % 100 in hash_list)
