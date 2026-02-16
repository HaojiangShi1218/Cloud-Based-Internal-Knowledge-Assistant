import os
import boto3
from typing import Any, Dict, List, Optional

from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth


def _client() -> OpenSearch:
    host = os.getenv("OPENSEARCH_HOST", "")
    if not host:
        raise RuntimeError("OPENSEARCH_HOST is not set")

    region = os.getenv("AWS_REGION", "us-east-1")
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
        hosts=[{"host": host, "port": 443}],
        http_auth=awsauth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        timeout=60,
        max_retries=3,
        retry_on_timeout=True,
    )


def bulk_index_chunks(index: str, docs: List[Dict[str, Any]]) -> None:
    """
    docs items must include:
      - doc_id (str)
      - chunk_id (int)
      - text (str)
      - embedding (List[float], len=384)
    plus any metadata fields you want (page, source, title, ...)
    """
    client = _client()

    # Bulk format: action line + source line
    body: List[Dict[str, Any]] = []
    for d in docs:
        doc_id = d.get("doc_id", "doc")
        chunk_id = int(d.get("chunk_id", 0))
        _id = f"{doc_id}:{chunk_id}"

        body.append({"index": {"_index": index, "_id": _id}})
        body.append(d)

    resp = client.bulk(body=body, refresh=True)
    if resp.get("errors"):
        # surface first few failures to make debugging obvious
        items = resp.get("items", [])[:5]
        raise RuntimeError(f"OpenSearch bulk errors. Sample items: {items}")


def knn_search(index: str, query_vec: List[float], k: int) -> List[Dict[str, Any]]:
    client = _client()
    q = {
        "size": k,
        "_source": ["doc_id", "chunk_id", "text", "page", "source", "title"],
        "query": {
            "knn": {
                "embedding": {
                    "vector": query_vec,
                    "k": k
                }
            }
        }
    }

    resp = client.search(index=index, body=q)
    hits = resp.get("hits", {}).get("hits", [])
    out = []
    for h in hits:
        s = h.get("_source", {})
        s["_score"] = h.get("_score")
        out.append(s)
    return out
