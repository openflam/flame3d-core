import re
import json
from collections import defaultdict
from typing import Any

import psycopg2
import psycopg2.extensions
from psycopg2.extras import RealDictCursor

from server.database.connection import DATABASE_URL
from server.database.spatial_tables import _linestring_z

def _pg_conn() -> psycopg2.extensions.connection:
    """Open a Postgres/PostGIS connection to the flame3d-core database."""
    return psycopg2.connect(DATABASE_URL)

def get_table_name(dataset_name: str) -> str:
    """Return a safe PostgreSQL identifier for a dataset table name."""
    return re.sub(r"[^a-z0-9_]", "_", dataset_name.lower())

def check_dataset_exists(dataset_name: str) -> bool:
    table = get_table_name(dataset_name)
    con = _pg_conn()
    try:
        with con.cursor() as cur:
            cur.execute(
                "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = %s)",
                (table,)
            )
            return cur.fetchone()[0]
    finally:
        con.close()

def list_dataset_tables() -> list[str]:
    """Return dataset table names from the PostGIS database."""
    con = _pg_conn()
    try:
        with con.cursor() as cur:
            cur.execute(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                  AND table_type = 'BASE TABLE'
                ORDER BY table_name
                """
            )
            return [row[0] for row in cur.fetchall()]
    finally:
        con.close()

def fetch_all_components(dataset_name: str) -> list[dict]:
    """Fetch all components fresh from the database."""
    table = get_table_name(dataset_name)
    con = _pg_conn()
    try:
        with con.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(f'SELECT component_id, caption, bbox_json, best_crop FROM "{table}" ORDER BY component_id')
            return cur.fetchall()
    finally:
        con.close()

def fetch_components_by_ids(dataset_name: str, component_ids: list[int]) -> list[dict]:
    if not component_ids:
        return []
    table = get_table_name(dataset_name)
    con = _pg_conn()
    try:
        with con.cursor(cursor_factory=RealDictCursor) as cur:
            placeholders = ",".join(["%s"] * len(component_ids))
            cur.execute(
                f'SELECT component_id, caption, bbox_json, best_crop FROM "{table}" WHERE component_id IN ({placeholders})',
                tuple(component_ids)
            )
            return cur.fetchall()
    finally:
        con.close()

def fetch_components_in_radius(dataset_name: str, component_id: int, radius: float) -> list[dict]:
    """Fetch components within a given radius of a target component using 3D spatial index."""
    table = get_table_name(dataset_name)
    con = _pg_conn()
    try:
        with con.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                f'''
                SELECT c2.component_id, c2.caption, c2.bbox_json, c2.best_crop
                FROM "{table}" c1
                JOIN "{table}" c2
                ON ST_3DDWithin(c1.bbox_geom, c2.bbox_geom, %s)
                WHERE c1.component_id = %s AND c2.component_id != %s
                ''',
                (radius, component_id, component_id)
            )
            return cur.fetchall()
    finally:
        con.close()

def fetch_first_component(dataset_name: str) -> dict | None:
    table = get_table_name(dataset_name)
    con = _pg_conn()
    try:
        with con.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                f'SELECT component_id, caption, bbox_json, best_crop FROM "{table}" ORDER BY component_id LIMIT 1'
            )
            return cur.fetchone()
    finally:
        con.close()

def fetch_component_info(dataset_name: str, component_id: int) -> dict | None:
    table = get_table_name(dataset_name)
    con = _pg_conn()
    try:
        with con.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                f'SELECT component_id, caption, bbox_json, best_crop FROM "{table}" WHERE component_id = %s',
                (component_id,)
            )
            return cur.fetchone()
    finally:
        con.close()

def update_component(dataset_name: str, component_id: int, caption: str | None, bbox: dict | None) -> int:
    table = get_table_name(dataset_name)
    fields = []
    params = []
    if caption is not None:
        fields.append("caption = %s")
        params.append(caption)
    if bbox is not None:
        fields.append("bbox_json = %s")
        params.append(json.dumps(bbox))
        
        geom_wkt = _linestring_z(bbox)
        if geom_wkt:
            fields.append("bbox_geom = ST_GeomFromText(%s, 0)")
            params.append(geom_wkt)
    
    if not fields:
        return 0
        
    params.append(component_id)
    
    con = _pg_conn()
    try:
        with con.cursor() as cur:
            cur.execute(
                f'UPDATE "{table}" SET {", ".join(fields)} WHERE component_id = %s',
                tuple(params)
            )
            con.commit()
            return cur.rowcount
    finally:
        con.close()

def get_next_component_id(dataset_name: str) -> int:
    table = get_table_name(dataset_name)
    con = _pg_conn()
    try:
        with con.cursor() as cur:
            cur.execute(f'SELECT COALESCE(MAX(component_id), 0) + 1 FROM "{table}"')
            return cur.fetchone()[0]
    finally:
        con.close()

def add_row(dataset_name: str, component_id: int, caption: str, bbox: dict, best_crop: str) -> int:
    table = get_table_name(dataset_name)
    con = _pg_conn()
    try:
        with con.cursor() as cur:
            bbox_json = json.dumps(bbox)
            geom_wkt = _linestring_z(bbox)
            if geom_wkt:
                cur.execute(
                    f'''
                    INSERT INTO "{table}" (component_id, caption, bbox_json, best_crop, bbox_geom)
                    VALUES (%s, %s, %s, %s, ST_GeomFromText(%s, 0))
                    ''',
                    (component_id, caption, bbox_json, best_crop, geom_wkt)
                )
            else:
                cur.execute(
                    f'''
                    INSERT INTO "{table}" (component_id, caption, bbox_json, best_crop)
                    VALUES (%s, %s, %s, %s)
                    ''',
                    (component_id, caption, bbox_json, best_crop)
                )
            con.commit()
            return cur.rowcount
    finally:
        con.close()

def delete_component(dataset_name: str, component_id: int) -> int:
    table = get_table_name(dataset_name)
    con = _pg_conn()
    try:
        with con.cursor() as cur:
            cur.execute(f'DELETE FROM "{table}" WHERE component_id = %s', (component_id,))
            con.commit()
            return cur.rowcount
    finally:
        con.close()

def search_components_tsvector(dataset_name: str, prefilter_query: str, prefilter_limit: int = 1000, candidate_ids: list[int] | None = None) -> list[tuple]:
    table = get_table_name(dataset_name)
    con = _pg_conn()
    
    # Process prefilter_query to replace spaces with ' OR ' for PostGIS websearch_to_tsquery
    parts = [p.strip() for p in prefilter_query.split() if p.strip()]
    or_query = " OR ".join(parts)
    
    where_clauses = ["to_tsvector('english', COALESCE(caption, '')) @@ websearch_to_tsquery('english', %s)"]
    params = [or_query]
    
    if candidate_ids is not None:
        if not candidate_ids:
            return []
        where_clauses.append("component_id = ANY(%s)")
        params.append(candidate_ids)
        
    where_str = " AND ".join(where_clauses)
    params.append(prefilter_limit)
    
    try:
        with con.cursor() as cur:
            cur.execute(
                f'''
                SELECT component_id, caption, bbox_json, best_crop
                FROM "{table}"
                WHERE {where_str}
                ORDER BY component_id
                LIMIT %s
                ''',
                tuple(params),
            )
            return cur.fetchall()
    finally:
        con.close()

def bm25_search(
    dataset_name: str,
    search_terms: str | list[str],
    top_k: int = 10,
    apply_elbow: bool = False,
    gap_threshold: float = 1.0,
    ratio_threshold: float = 0.7,
    candidate_ids: list[int] | None = None,
) -> dict[str, Any]:
    import bm25s
    import Stemmer
    
    if isinstance(search_terms, str):
        cleaned_terms = [search_terms.strip()] if search_terms.strip() else []
        prefilter_query = search_terms.strip()
    else:
        cleaned_terms = [
            term.strip() for term in search_terms if isinstance(term, str) and term.strip()
        ]
        prefilter_query = " ".join(cleaned_terms)

    if not cleaned_terms:
        return {
            "dataset_name": dataset_name,
            "search_terms": cleaned_terms,
            "results": [],
            "component_ids": [],
            "reason": "No valid search terms were provided."
        }
        
    rows = search_components_tsvector(dataset_name, prefilter_query, prefilter_limit=1000, candidate_ids=candidate_ids)
    
    if not rows:
        return {
            "dataset_name": dataset_name,
            "search_terms": cleaned_terms,
            "results": [],
            "component_ids": [],
            "reason": "No components matched the prefilter query."
        }

    component_ids = [int(row[0]) for row in rows]
    captions = [row[1] or "" for row in rows]

    stemmer = Stemmer.Stemmer("english")
    corpus_tokens = bm25s.tokenize(captions, stopwords="en", stemmer=stemmer)
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)

    score_sum = defaultdict(float)
    matched_terms = defaultdict(list)

    k_each = min(max(1, top_k), len(captions))
    
    if len(cleaned_terms) == 1 and apply_elbow:
        query_tokens = bm25s.tokenize(cleaned_terms[0], stemmer=stemmer)
        results, scores = retriever.retrieve(query_tokens, k=k_each)
        n = results.shape[1]
        keep = [0] if n > 0 else []
        for i in range(1, n):
            prev_score = scores[0, i - 1]
            cur_score = scores[0, i]
            if cur_score <= 0:
                break
            if (prev_score - cur_score) >= gap_threshold or (cur_score / max(prev_score, 1e-9)) <= ratio_threshold:
                break
            keep.append(i)
        
        for i in keep:
            doc_idx = int(results[0, i])
            score = float(scores[0, i])
            if score > 0:
                score_sum[doc_idx] += score
                matched_terms[doc_idx].append(cleaned_terms[0])
    else:
        for term in cleaned_terms:
            query_tokens = bm25s.tokenize(term, stemmer=stemmer)
            results, scores = retriever.retrieve(query_tokens, k=k_each)

            for i in range(results.shape[1]):
                doc_idx = int(results[0, i])
                score = float(scores[0, i])
                if score <= 0:
                    continue
                score_sum[doc_idx] += score
                matched_terms[doc_idx].append(term)

    ranked_doc_indices = sorted(
        score_sum.keys(), key=lambda idx: score_sum[idx], reverse=True
    )[: max(1, top_k)]

    results_payload = []
    matched_component_ids = []
    
    for doc_idx in ranked_doc_indices:
        _, caption, bbox_json, best_crop = rows[doc_idx]
        try:
            bbox = json.loads(bbox_json) if bbox_json else {}
        except json.JSONDecodeError:
            bbox = {}

        cid = component_ids[doc_idx]
        matched_component_ids.append(cid)
        results_payload.append(
            {
                "component_id": cid,
                "caption": caption or "",
                "best_crop": best_crop,
                "bbox": bbox,
                "score": score_sum[doc_idx],
                "matched_terms": matched_terms[doc_idx],
            }
        )

    reason = (
        f"BM25 matched {len(results_payload)} component(s) from {len(rows)} candidate caption(s) "
        f"using prefilter."
    )

    return {
        "dataset_name": dataset_name,
        "search_terms": cleaned_terms,
        "results": results_payload,
        "component_ids": matched_component_ids,
        "reason": reason,
    }
