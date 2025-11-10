from collections import defaultdict
from ..types import ScoredDoc

# def rrf(lists: list[list[ScoredDoc]], k_const: int = 60, limit: int | None = None) -> list[ScoredDoc]:
#     agg = defaultdict(float)  # doc_id -> fused score
#     best_rank = {}            # keep a representative rank
#     for ranklist in lists:
#         for d in ranklist:
#             agg[d.doc_id] += 1.0 / (k_const + d.rank)
#             best_rank[d.doc_id] = min(best_rank.get(d.doc_id, d.rank), d.rank)

#     fused = [ScoredDoc(doc_id=doc, score=score, rank=best_rank[doc]) for doc, score in agg.items()]
#     fused.sort(key=lambda x: x.score, reverse=True)

#     return fused if limit is None else fused[:limit]

def rrf(lists: list[list["ScoredDoc"]], k_const: int = 60, limit: int | None = None) -> list["ScoredDoc"]:
    # --- Before fusion ---
    # print("\n=== BEFORE FUSION ===")
    # for i, ranklist in enumerate(lists):
    #     print(f"List {i+1}:")
        # for d in ranklist:
        #     print(f"  id={d.doc_id:<10} rank={d.rank:<3} score={getattr(d, 'score', None)}")

    # --- Fuse scores ---
    agg = defaultdict(float)
    best_rank = {}
    for ranklist in lists:
        for d in ranklist:
            agg[d.doc_id] += 1.0 / (k_const + d.rank)
            best_rank[d.doc_id] = min(best_rank.get(d.doc_id, d.rank), d.rank)

    fused = [ScoredDoc(doc_id=doc, score=score, rank=best_rank[doc]) for doc, score in agg.items()]
    fused.sort(key=lambda x: x.score, reverse=True)

    # --- After fusion ---
    # print("\n=== AFTER FUSION ===")
    # for i, d in enumerate(fused[:limit] if limit else fused, start=1):
    #     print(f"Rank {i:<3} id={d.doc_id:<10} fused_score={d.score:.5f} best_rank={d.rank}")

    return fused if limit is None else fused[:limit]
