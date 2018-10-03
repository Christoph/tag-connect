

def word_sets(texts, ids):
    sets = [set(t.split(" ")) for t in texts]

    group = [sets[i] for i in ids]
    # rest = [s for s in sets if s not in group]

    intersection = group[0].intersection(*group[1:])

    diff = {}

    for i, id in enumerate(ids):
        d = group[i].difference(*[x for j, x in enumerate(group) if j != i])
        diff[id] = d

    out = {
        "inter": intersection,
        "diffs": diff
    }

    return out

def group_comp(texts, ids1, ids2):
    sets = [set(t.split(" ")) for t in texts]

    group1 = [sets[i] for i in ids1]
    g1 = group1[0].union(*group1[1:])
    group2 = [sets[i] for i in ids2]
    g2 = group2[0].union(*group2[1:])
    # rest = [s for s in sets if s not in group1+group2]
    # r = rest[0].intersection(*rest[1:])

    inter1 = group1[0].intersection(*group1[1:])
    inter2 = group2[0].intersection(*group2[1:])
    inter12 = g1.intersection(g2)

    diff12 = g1.difference(g2)
    diff21 = g2.difference(g1)

    out = {
        "inter1": inter1,
        "inter2": inter2,
        "inter12": inter12,
        "diff12": diff12,
        "diff21": diff21,
    }

    return out
