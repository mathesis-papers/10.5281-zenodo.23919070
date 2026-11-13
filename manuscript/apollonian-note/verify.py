import argparse
import json
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import scipy
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh


def build_stages(last_stage):
    births = [0, 0, 0, 0]
    edges = list(combinations(range(4), 2))
    faces = [(3, 0, 1), (3, 1, 2), (3, 2, 0)]
    for stage in range(last_stage + 1):
        vertex_count = len(births)
        rows = [first for first, second in edges] + [second for first, second in edges]
        columns = [second for first, second in edges] + [first for first, second in edges]
        adjacency = coo_matrix((np.ones(len(rows), dtype=np.int64), (rows, columns)),
                               shape=(vertex_count, vertex_count)).tocsr()
        yield stage, adjacency, tuple(births), tuple(faces)
        next_faces = []
        for face in faces:
            centre = len(births)
            births.append(stage + 1)
            edges.extend((centre, corner) for corner in face)
            for position in range(3):
                corners = list(face)
                corners[position] = centre
                next_faces.append(tuple(corners))
        faces = next_faces


def ordered_weights(cutoff):
    weights = [3.0, 2.0, 2.0, 2.0]
    for generation in range(1, cutoff + 1):
        weights.extend([3.0 / 2**generation] * 3**generation)
    return weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-stage', type=int, default=8)
    parser.add_argument('--output', type=Path, default=Path('checks.json'))
    args = parser.parse_args()
    if not 1 <= args.max_stage <= 9:
        parser.error('Choose a maximum stage between 1 and 9')
    snapshots = {}
    records = []
    gram_checks = 0
    for stage, adjacency, births, faces in build_stages(args.max_stage):
        vertex_count = adjacency.shape[0]
        assert vertex_count == (3**(stage + 1) + 5) // 2
        assert len(faces) == 3**(stage + 1)
        assert adjacency.nnz == 2 * (3 * vertex_count - 6)
        assert (adjacency != adjacency.T).nnz == 0
        assert not adjacency.diagonal().any()
        degrees = np.asarray(adjacency.sum(axis=1)).ravel()
        face_counts = np.zeros(vertex_count, dtype=np.int64)
        pair_counts = Counter()
        for face in faces:
            assert len(set(face)) == 3
            for vertex in face:
                face_counts[vertex] += 1
            for first, second in combinations(sorted(face), 2):
                assert adjacency[first, second] == 1
                pair_counts[first, second] += 1
        assert max(pair_counts.values()) <= 2
        for vertex, birth in enumerate(births):
            expected = 2**(stage + 1) + 1 if vertex < 3 else 3 * 2**(stage - birth)
            assert degrees[vertex] == expected
            assert face_counts[vertex] == expected - (1 if vertex < 3 else 0)
        for cutoff, (old_adjacency, old_face_counts, old_pair_counts) in snapshots.items():
            old_count = old_adjacency.shape[0]
            block = adjacency[:old_count, old_count:]
            gram = (block @ block.T).toarray()
            expected = np.diag((2**(stage - cutoff) - 1) * old_face_counts)
            for (first, second), multiplicity in old_pair_counts.items():
                expected[first, second] = expected[second, first] = (stage - cutoff) * multiplicity
            assert np.array_equal(gram, expected)
            weights = np.concatenate(([2.0, 2.0, 2.0, 3.0],
                                      [3.0 / 2**birth for birth in births[4:old_count]]))
            difference = gram / 2**stage - np.diag(weights)
            gram_bound = 3 * (1 + 2 * (stage - cutoff)) * 2.0**(cutoff - stage)
            assert np.max(np.abs(np.linalg.eigvalsh(difference))) <= gram_bound + 1e-12
            directed = adjacency[old_count:, old_count:].tocoo()
            row_degrees = np.zeros(vertex_count - old_count, dtype=np.int64)
            column_degrees = np.zeros_like(row_degrees)
            for first, second in zip(directed.row, directed.col):
                if births[first + old_count] > births[second + old_count]:
                    row_degrees[first] += 1
                    column_degrees[second] += 1
                elif first < second:
                    assert births[first + old_count] != births[second + old_count]
            assert row_degrees.max(initial=0) <= 3
            assert column_degrees.max(initial=0) <= 3 * 2**(stage - cutoff - 1)
            gram_checks += 1
        if stage <= 3:
            snapshots[stage] = (adjacency.copy(), face_counts.copy(), pair_counts.copy())
        floating = adjacency.astype(float)
        eigen_count = min(8, vertex_count - 1)
        start = np.linspace(1.0, 2.0, vertex_count)
        largest, vectors = eigsh(floating, k=eigen_count, which='LA', tol=1e-11,
                                  v0=start, maxiter=10000)
        smallest, negative_vectors = eigsh(floating, k=eigen_count, which='SA', tol=1e-11,
                                           v0=start, maxiter=10000)
        residual = max(np.linalg.norm(floating @ vectors - vectors * largest, axis=0).max(),
                       np.linalg.norm(floating @ negative_vectors - negative_vectors * smallest,
                                      axis=0).max())
        assert residual < 1e-7
        normalized = largest[::-1] / 2**(stage / 2)
        negative_normalized = smallest / 2**(stage / 2)
        for cutoff in range(min(3, stage - 1) + 1):
            weights = ordered_weights(cutoff)
            ranks = min(eigen_count, len(weights))
            remainder = max(6 * 2**(-(stage - cutoff) / 2),
                            3 * np.sqrt(2) * 2**(-cutoff / 2))
            gram_bound = 3 * (1 + 2 * (stage - cutoff)) * 2.0**(cutoff - stage)
            for index in range(ranks):
                limit = np.sqrt(weights[index])
                error_bound = remainder + gram_bound / limit
                assert abs(normalized[index] - limit) <= error_bound + 1e-10
                assert abs(negative_normalized[index] + limit) <= error_bound + 1e-10
        if stage == 0:
            assert np.allclose(largest, [-1, -1, 3], atol=1e-10)
        records.append({'stage': stage, 'vertices': vertex_count,
                        'edges': adjacency.nnz // 2, 'inner_faces': len(faces),
                        'largest_normalized': normalized.tolist(),
                        'smallest_normalized': negative_normalized.tolist(),
                        'largest_eigenpair_residual': float(residual)})
    output = {'schema_version': 1,
              'coverage': {'exact_integer_checks': 'Graph counts, all vertex degrees, face incidence, exact cutoff Gram identities, and late-edge orientation bounds.',
                           'stage_range': [0, args.max_stage], 'cutoff_range': [0, 3],
                           'cutoff_stage_pairs': gram_checks,
                           'floating_point_checks': 'Up to eight largest and smallest adjacency eigenvalues, with eigenpair residuals.',
                           'infinite_limits': 'The infinite limits follow from the written proof in note.pdf.'},
              'software': {'numpy': np.__version__, 'scipy': scipy.__version__},
              'limiting_weights_first_eight': ordered_weights(2)[:8],
              'stages': records}
    args.output.write_text(json.dumps(output, indent=2) + '\n')
    print(f'Exact checks passed for stages 0 through {args.max_stage}; {gram_checks} cutoff-stage pairs.')
    print(f'Eigenpair residual maximum: {max(record["largest_eigenpair_residual"] for record in records):.3e}')


if __name__ == '__main__':
    main()
