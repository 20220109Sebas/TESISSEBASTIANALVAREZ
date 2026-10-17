"""Resume a fixed annex queue without retrying failed or oversized files implicitly."""


def next_annex_batch(queue, accesses, *, limit=5):
    if not 1 <= limit <= 5:
        raise ValueError("A bounded batch contains one to five documents")
    if len({d["opportunity_id"] for d in queue}) != len(queue) or len({d["url"] for d in queue}) != len(queue):
        raise ValueError("Duplicate opportunity or URL in fixed queue")
    attempted = {r["url"] for r in accesses}
    completed = {r["url"] for r in accesses if r["status"] == "downloaded_pdf"}
    queue_urls = {d["url"] for d in queue}
    pending = [d for d in queue if d["url"] not in completed]
    untouched = [d for d in queue if d["url"] not in attempted]
    return {"selected": untouched[:limit], "completed_opportunities": len(completed & queue_urls),
            "pending_opportunities": len(pending), "unattempted_opportunities": len(untouched),
            "pending_previously_attempted": [d for d in pending if d["url"] in attempted],
            "automatic_retries": False}
