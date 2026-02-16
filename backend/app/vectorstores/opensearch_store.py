import boto3
from typing import Any, Dict, List

from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
from app.config import settings


def get_os_client() -> OpenSearch:
    host = settings.OPENSEARCH_HOST
    if not host:
        raise RuntimeError("OPENSEARCH_HOST is not set")

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
        hosts=[{"host": host, "port": 443}],
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
        "_source": ["doc_id", "chunk_id", "text", "page", "source", "title"],
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
                "doc_id": s.get("doc_id", ""),
                "chunk_id": s.get("chunk_id"),
                "title": s.get("title", ""),
                "score": float(h.get("_score", 0.0)),
            }
        )
    return out
