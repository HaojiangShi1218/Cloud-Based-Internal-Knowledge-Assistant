import boto3
from typing import Any, Dict, List
from urllib.parse import urlparse

from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
from app.config import settings


def _parse_host(host: str) -> Dict[str, Any]:
    """
    Supports:
      - "opensearch"                -> http://opensearch:9200
      - "opensearch:9200"           -> http://opensearch:9200
      - "http://opensearch:9200"    -> http://opensearch:9200
      - "https://vpc-xxx.es.amazonaws.com" -> https://...:443
    """
    host = host.strip()
    if "://" not in host:
        host = "http://" + host

    u = urlparse(host)
    use_ssl = (u.scheme == "https")
    port = u.port or (443 if use_ssl else 9200)
    return {"hostname": u.hostname, "port": port, "use_ssl": use_ssl}


def get_os_client() -> OpenSearch:
    raw = settings.OPENSEARCH_HOST
    if not raw:
        raise RuntimeError("OPENSEARCH_HOST is not set")

    p = _parse_host(raw)

    # Local docker (no SigV4, no TLS)
    if not p["use_ssl"] and p["port"] == 9200:
        return OpenSearch(
            hosts=[{"host": p["hostname"], "port": p["port"]}],
            use_ssl=False,
            verify_certs=False,
            timeout=60,
            max_retries=3,
            retry_on_timeout=True,
        )

    # AWS-managed (SigV4 + TLS)
    region = settings.AWS_REGION
    service = "es"

    session = boto3.Session()
    creds = session.get_credentials()
    if creds is None:
        raise RuntimeError("No AWS credentials available (EC2 role / IMDS)")

    awsauth = AWS4Auth(
        creds.access_key,
        creds.secret_key,
        region,
        service,
        session_token=creds.token,
    )

    return OpenSearch(
        hosts=[{"host": p["hostname"], "port": p["port"]}],
        http_auth=awsauth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        timeout=60,
        max_retries=3,
        retry_on_timeout=True,
    )


def knn_search(query_vector: List[float], k: int) -> List[Dict[str, Any]]:
    client = get_os_client()
    q = {
        "size": k,
        "_source": ["doc_id", "chunk_id", "doc_chunk_seq", "text", "page", "page_end", "source", "title"],
        "query": {
            "knn": {
                "embedding": {
                    "vector": query_vector,
                    "k": k
                }
            }
        }
    }

    resp = client.search(index=settings.OPENSEARCH_INDEX, body=q)
    hits = resp.get("hits", {}).get("hits", [])
    out = []
    for h in hits:
        s = h.get("_source", {})
        out.append(
            {
                "text": s.get("text", ""),
                "source": s.get("source", ""),
                "page": s.get("page"),
                "page_end": s.get("page_end", s.get("page")),
                "doc_id": s.get("doc_id", ""),
                "chunk_id": s.get("chunk_id"),
                "doc_chunk_seq": s.get("doc_chunk_seq"),
                "title": s.get("title", ""),
                "score": float(h.get("_score") or 0.0),
            }
        )
    return out


def fetch_doc_seq_chunks(doc_id: str, seq_values: List[int]) -> List[Dict[str, Any]]:
    if not doc_id or not seq_values:
        return []

    client = get_os_client()
    q = {
        "size": len(seq_values),
        "_source": ["doc_id", "chunk_id", "doc_chunk_seq", "text", "page", "page_end", "source", "title"],
        "query": {
            "bool": {
                "must": [
                    {"term": {"doc_id": doc_id}},
                    {"terms": {"doc_chunk_seq": seq_values}},
                ]
            }
        },
        "sort": [{"doc_chunk_seq": {"order": "asc"}}],
    }

    resp = client.search(index=settings.OPENSEARCH_INDEX, body=q)
    hits = resp.get("hits", {}).get("hits", [])
    out: List[Dict[str, Any]] = []
    for h in hits:
        s = h.get("_source", {})
        out.append(
            {
                "text": s.get("text", ""),
                "source": s.get("source", ""),
                "page": s.get("page"),
                "page_end": s.get("page_end", s.get("page")),
                "doc_id": s.get("doc_id", ""),
                "chunk_id": s.get("chunk_id"),
                "doc_chunk_seq": s.get("doc_chunk_seq"),
                "title": s.get("title", ""),
                "score": float(h.get("_score") or 0.0),
            }
        )
    return out
