#!/usr/bin/env python3
"""Refresh the RWKV tracker using only the Python standard library."""

import html
import json
import os
import re
import time
import xml.etree.ElementTree as ET
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen

GITHUB_SOURCE = "https://github.com/search?o=desc&p=1&q=rwkv&s=updated&type=Repositories"
AGENT_SOURCE = "https://agent.objects.rwkvos.com/en"
AGENT_FEED = "https://agent.objects.rwkvos.com/agent-data.json"
HF_SOURCE = "https://huggingface.co/models?sort=modified&search=rwkv"
ARXIV_SOURCE = "https://arxiv.org/search/?query=RWKV&searchtype=all"
ARXIV_QUERY = (
    'ti:RWKV* OR abs:RWKV* '
    'OR ti:"Receptance Weighted Key Value" '
    'OR abs:"Receptance Weighted Key Value"'
)
MAIN_MODEL = "BlinkDL/rwkv7-g1"
USER_AGENT = "rwkv-projects-tracking/1.0"
HISTORY_DIR = Path("history")
HISTORY_BACKFILL_DAYS = 7


def parse_time(value):
    if not value:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def iso_time(value):
    parsed = parse_time(value)
    return parsed.isoformat().replace("+00:00", "Z") if parsed else None


def rate_delay(headers):
    retry_after = headers.get("Retry-After")
    if retry_after and retry_after.isdigit():
        return int(retry_after)
    reset = headers.get("X-RateLimit-Reset")
    if reset and reset.isdigit():
        return max(1, int(reset) - int(time.time()) + 1)
    return 60


def fetch_json(url, *, github=False):
    headers = {"Accept": "application/json", "User-Agent": USER_AGENT}
    token = os.environ.get("GITHUB_TOKEN") if github else None
    if token:
        headers["Authorization"] = f"Bearer {token}"
    for attempt in range(3):
        try:
            with urlopen(Request(url, headers=headers), timeout=60) as response:
                payload = json.load(response)
                response_headers = response.headers
            if github and response_headers.get("X-RateLimit-Remaining") == "0":
                time.sleep(rate_delay(response_headers))
            return payload, response_headers
        except HTTPError as error:
            limited = error.code == 429 or (
                error.code == 403 and error.headers.get("X-RateLimit-Remaining") == "0"
            )
            if not limited or attempt == 2:
                raise
            time.sleep(rate_delay(error.headers))
    raise AssertionError("unreachable")


def github_page(query, page):
    params = urlencode({
        "q": query,
        "sort": "updated",
        "order": "desc",
        "per_page": 100,
        "page": page,
    })
    payload, _ = fetch_json(
        f"https://api.github.com/search/repositories?{params}", github=True
    )
    if payload.get("incomplete_results"):
        raise RuntimeError(f"GitHub returned incomplete results for {query}")
    return payload


def github_shard(start, end):
    """Return a half-open creation-date shard, splitting at GitHub's 1,000-result cap."""
    last_day = end - timedelta(days=1)
    query = f"rwkv is:public created:{start}..{last_day}"
    first = github_page(query, 1)
    total = first["total_count"]
    if total > 1000:
        days = (end - start).days
        if days < 2:
            raise RuntimeError(f"More than 1,000 repositories were created on {start}")
        middle = start + timedelta(days=days // 2)
        return github_shard(start, middle) + github_shard(middle, end)

    rows = list(first["items"])
    for page in range(2, (total + 99) // 100 + 1):
        rows.extend(github_page(query, page)["items"])
    identities = [row["id"] for row in rows]
    if len(rows) != total or len(set(identities)) != total:
        raise RuntimeError(f"GitHub results changed during pagination for {query}; rerun the updater")
    return rows


def get_repositories(now):
    # The 2024 boundary avoids the aggregate 1,000-result cap; shards split further as needed.
    rows = github_shard(date(2008, 1, 1), date(2024, 1, 1))
    rows += github_shard(date(2024, 1, 1), now.date() + timedelta(days=1))
    projects = [{
        "kind": "repository",
        "id": row["full_name"],
        "name": row["full_name"],
        "url": row["html_url"],
        "description": row["description"] or "",
        "updated_at": iso_time(row["updated_at"]),
        "pushed_at": iso_time(row["pushed_at"]),
        "created_at": iso_time(row["created_at"]),
        "stars": row["stargazers_count"],
        "forks": row["forks_count"],
        "language": row["language"],
        "archived": row["archived"],
    } for row in rows]
    return unique_sorted(projects)


def safe_url(value):
    parsed = urlsplit(value)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError(f"Invalid project URL: {value!r}")
    return value


def agent_identity(url):
    parsed = urlsplit(safe_url(url))
    host = parsed.hostname.lower().removeprefix("www.")
    path = parsed.path.rstrip("/")
    if path.lower().endswith(".git"):
        path = path[:-4]
    return (host + path).casefold()


def english(value):
    if value is None:
        return ""
    if not isinstance(value, dict) or not isinstance(value.get("en"), str):
        raise ValueError("Agent catalog returned an invalid localized field")
    return value["en"].strip()


def get_agents():
    rows, _ = fetch_json(AGENT_FEED)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Agent catalog must be a non-empty array")
    projects = []
    for row in rows:
        name = row["项目名称"].strip()
        url = safe_url(row["GitHub地址"].strip())
        projects.append({
            "kind": "agent",
            "id": agent_identity(url),
            "name": name,
            "url": url,
            "description": english(row.get("介绍")),
            "category": english(row.get("项目类型")),
            "classification": english(row.get("分类结论")),
            "owner": row.get("创建用户名"),
            "updated_at": iso_time(row.get("最后更新时间")),
            "created_at": iso_time(row.get("创建时间")),
            "stars": row.get("Stars", 0),
            "forks": row.get("Fork", 0),
            "language": row.get("主要语言"),
            "license": row.get("许可证"),
        })
    return unique_sorted(projects)


def next_link(headers):
    for header in headers.get_all("Link", []):
        for part in header.split(","):
            if 'rel="next"' in part:
                return part.split("<", 1)[1].split(">", 1)[0]
    return None


def model_url(model_id):
    return f"https://huggingface.co/{model_id}"


def model_project(row):
    model_id = row["id"]
    return {
        "kind": "model",
        "id": model_id,
        "name": model_id,
        "url": model_url(model_id),
        "description": "",
        "updated_at": iso_time(row.get("lastModified")),
        "created_at": iso_time(row.get("createdAt")),
        "downloads": row.get("downloads", 0),
        "likes": row.get("likes", 0),
        "pipeline_tag": row.get("pipeline_tag"),
        "main": model_id == MAIN_MODEL,
    }


def get_models():
    expands = ["author", "createdAt", "downloads", "likes", "lastModified", "pipeline_tag"]
    params = [("search", "rwkv"), ("sort", "lastModified"), ("direction", "-1"), ("limit", "100")]
    params += [("expand", field) for field in expands]
    url = f"https://huggingface.co/api/models?{urlencode(params)}"
    projects = []
    while url:
        rows, headers = fetch_json(url)
        if not isinstance(rows, list):
            raise ValueError("Hugging Face model search must return an array")
        projects.extend(model_project(row) for row in rows)
        url = next_link(headers)
        if url:
            parsed = urlsplit(url)
            if parsed.scheme != "https" or parsed.hostname != "huggingface.co":
                raise ValueError("Hugging Face returned an invalid pagination URL")
    return unique_sorted(projects, deduplicate=True)


def get_main_model(now):
    expands = ["lastModified", "downloads", "likes", "pipeline_tag", "siblings"]
    params = urlencode([("expand", field) for field in expands])
    detail, _ = fetch_json(f"https://huggingface.co/api/models/{MAIN_MODEL}?{params}")

    checkpoint_dates = []
    for sibling in detail.get("siblings", []):
        filename = sibling.get("rfilename", "")
        if filename.endswith(".pth"):
            for match in re.findall(r"(?<!\d)(20\d{6})(?!\d)", filename):
                try:
                    checkpoint_dates.append(datetime.strptime(match, "%Y%m%d").date())
                except ValueError:
                    pass

    latest_commit = None
    page = 0
    while latest_commit is None:
        commits, _ = fetch_json(
            f"https://huggingface.co/api/models/{MAIN_MODEL}/commits/main?limit=100&p={page}"
        )
        latest_commit = next((row for row in commits if row.get("title") == "new model"), None)
        if len(commits) < 100:
            break
        page += 1

    commit_time = parse_time(latest_commit["date"]) if latest_commit else None
    fresh = commit_time is not None and timedelta(0) <= now - commit_time <= timedelta(days=31)
    return {
        "id": MAIN_MODEL,
        "url": model_url(MAIN_MODEL),
        "source_last_modified": iso_time(detail.get("lastModified")),
        "latest_new_model_commit_at": iso_time(latest_commit["date"]) if latest_commit else None,
        "new_model_commit_within_31_days": fresh,
        "latest_current_checkpoint_filename_date": max(checkpoint_dates).isoformat() if checkpoint_dates else None,
    }


def unique_sorted(projects, *, deduplicate=False):
    by_id = {}
    for project in projects:
        if project["id"] in by_id and not deduplicate:
            raise ValueError(f"Duplicate {project['kind']} identity: {project['id']}")
        by_id[project["id"]] = project
    return sorted(by_id.values(), key=lambda p: (p["updated_at"] or "", p["id"]), reverse=True)


def load_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def jsonl(records):
    return "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records)


def fetch_text(url):
    request = Request(url, headers={"Accept": "application/atom+xml", "User-Agent": USER_AGENT})
    with urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8")


def get_papers():
    namespaces = {
        "atom": "http://www.w3.org/2005/Atom",
        "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
        "arxiv": "http://arxiv.org/schemas/atom",
    }
    start = 0
    total = None
    papers = []
    while total is None or start < total:
        params = urlencode({
            "search_query": ARXIV_QUERY,
            "start": start,
            "max_results": 200,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        })
        root = ET.fromstring(fetch_text(f"https://export.arxiv.org/api/query?{params}"))
        total = int(root.findtext("opensearch:totalResults", namespaces=namespaces))
        start_index = int(root.findtext("opensearch:startIndex", namespaces=namespaces))
        if start_index != start or total <= 0:
            raise ValueError("arXiv returned an invalid result window")
        entries = root.findall("atom:entry", namespaces)
        if not entries:
            raise ValueError("arXiv returned an empty intermediate page")

        for entry in entries:
            versioned_id = entry.findtext("atom:id", namespaces=namespaces).split("/abs/", 1)[1]
            paper_id = re.sub(r"v\d+$", "", versioned_id)
            primary = entry.find("arxiv:primary_category", namespaces)
            papers.append({
                "kind": "paper",
                "id": paper_id,
                "name": " ".join(entry.findtext("atom:title", namespaces=namespaces).split()),
                "url": f"https://arxiv.org/abs/{paper_id}",
                "pdf_url": f"https://arxiv.org/pdf/{paper_id}",
                "description": " ".join(entry.findtext("atom:summary", namespaces=namespaces).split()),
                "published_at": iso_time(entry.findtext("atom:published", namespaces=namespaces)),
                "updated_at": iso_time(entry.findtext("atom:updated", namespaces=namespaces)),
                "authors": [
                    author.findtext("atom:name", namespaces=namespaces)
                    for author in entry.findall("atom:author", namespaces)
                ],
                "categories": [category.get("term") for category in entry.findall("atom:category", namespaces)],
                "primary_category": primary.get("term") if primary is not None else None,
                "versioned_id": versioned_id,
            })
        start += len(entries)
        if start < total:
            time.sleep(3)
    if len(papers) != total:
        raise ValueError(f"Expected {total} arXiv papers, received {len(papers)}")
    return unique_sorted(papers)


def paper_history(papers, now):
    records = {record["id"]: record for record in load_jsonl(HISTORY_DIR / "papers.jsonl")}
    for paper in papers:
        previous = records.get(paper["id"], {})
        records[paper["id"]] = {
            "id": paper["id"],
            "title": paper["name"],
            "published_at": paper["published_at"],
            "updated_at": paper["updated_at"],
            "first_seen_at": previous.get("first_seen_at", now.isoformat().replace("+00:00", "Z")),
        }
    return sorted(records.values(), key=lambda record: (record["published_at"], record["id"]))


def commit_history(repositories, now):
    state_path = HISTORY_DIR / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    history_start = parse_time(state.get("history_start")) or now - timedelta(days=HISTORY_BACKFILL_DAYS)
    last_success = parse_time(state.get("last_success_started_at")) or history_start
    current_ids = {repository["id"] for repository in repositories}
    # Absent repositories must be treated as new if they later return to the search.
    scanned = set(state.get("scanned_repositories", [])) & current_ids
    records = {
        (record["repo"], record["sha"]): record
        for record in load_jsonl(HISTORY_DIR / "commits.jsonl")
    }
    selected = []
    for repository in repositories:
        pushed = parse_time(repository["pushed_at"])
        if not pushed or pushed < history_start:
            continue
        if repository["id"] not in scanned or pushed >= last_success - timedelta(minutes=5):
            selected.append(repository)

    print(f"Collecting default-branch commits from {len(selected)} recently pushed repositories.")
    for repository in selected:
        params = urlencode({"since": iso_time(history_start.isoformat()), "per_page": 100})
        url = f"https://api.github.com/repos/{repository['id']}/commits?{params}"
        while url:
            try:
                rows, headers = fetch_json(url, github=True)
            except HTTPError as error:
                if error.code == 409:  # Empty repository.
                    break
                raise
            if not isinstance(rows, list):
                raise ValueError(f"GitHub returned invalid commits for {repository['id']}")
            for row in rows:
                key = (repository["id"], row["sha"])
                if key in records:
                    continue
                records[key] = {
                    "repo": repository["id"],
                    "sha": row["sha"],
                    "url": row["html_url"],
                    "message": (row["commit"]["message"].splitlines() or [""])[0],
                    "author_date": iso_time(row["commit"]["author"]["date"]),
                    "committer_date": iso_time(row["commit"]["committer"]["date"]),
                    "author_login": (row.get("author") or {}).get("login"),
                    "first_seen_at": now.isoformat().replace("+00:00", "Z"),
                }
            url = next_link(headers)
            if url:
                parsed = urlsplit(url)
                if parsed.scheme != "https" or parsed.hostname != "api.github.com":
                    raise ValueError("GitHub returned an invalid commit pagination URL")

    scanned.update(repository["id"] for repository in repositories)
    next_state = {
        "history_start": iso_time(history_start.isoformat()),
        "last_success_started_at": now.isoformat().replace("+00:00", "Z"),
        "scanned_repositories": sorted(scanned),
    }
    history = sorted(records.values(), key=lambda row: (row["committer_date"], row["repo"], row["sha"]))
    return history, next_state


def project_changes(previous_snapshot, current_projects, now):
    previous = {(p["kind"], p["id"]): p for p in previous_snapshot.get("projects", [])}
    current = {(p["kind"], p["id"]): p for p in current_projects}
    known_kinds = set(previous_snapshot.get("sources", {}))
    observed_at = now.isoformat().replace("+00:00", "Z")
    events = []
    for key, project in current.items():
        kind = key[0]
        old = previous.get(key)
        if old is None:
            # Introducing a new source is a baseline, not hundreds of false "new" events.
            if kind in known_kinds:
                events.append({
                    "date": now.date().isoformat(), "observed_at": observed_at,
                    "kind": kind, "id": project["id"], "name": project["name"],
                    "url": project["url"], "event": "added",
                })
            continue
        changes = {}
        for field in ("stars", "forks", "downloads", "likes", "updated_at"):
            if project.get(field) != old.get(field):
                changes[field] = {"from": old.get(field), "to": project.get(field)}
        if changes:
            events.append({
                "date": now.date().isoformat(), "observed_at": observed_at,
                "kind": kind, "id": project["id"], "name": project["name"],
                "url": project["url"], "event": "changed", "changes": changes,
            })
    for key, old in previous.items():
        if key not in current:
            events.append({
                "date": now.date().isoformat(), "observed_at": observed_at,
                "kind": old["kind"], "id": old["id"], "name": old["name"],
                "url": old["url"], "event": "removed",
            })

    existing = load_jsonl(HISTORY_DIR / "changes.jsonl")
    signatures = {json.dumps(event, sort_keys=True, ensure_ascii=False) for event in existing}
    for event in events:
        signature = json.dumps(event, sort_keys=True, ensure_ascii=False)
        if signature not in signatures:
            existing.append(event)
            signatures.add(signature)
    return sorted(existing, key=lambda row: (row["date"], row["kind"], row["id"], row["event"]))


def daily_history(projects, changes, history_start, now):
    records = [row for row in load_jsonl(HISTORY_DIR / "daily.jsonl") if row.get("date") != now.date().isoformat()]
    counts = {kind: sum(p["kind"] == kind for p in projects) for kind in ("repository", "agent", "model", "paper")}
    today = [row for row in changes if row["date"] == now.date().isoformat()]
    records.append({
        "date": now.date().isoformat(),
        "generated_at": now.isoformat().replace("+00:00", "Z"),
        "history_start": iso_time(history_start.isoformat()),
        "counts": counts,
        "changes": {event: sum(row["event"] == event for row in today) for event in ("added", "removed", "changed")},
    })
    return sorted(records, key=lambda row: row["date"])


def markdown(value):
    text = html.escape(str(value if value is not None else "—"))
    return (text.replace("|", "\\|").replace("[", "\\[").replace("]", "\\]")
            .replace("\r", " ").replace("\n", " "))


def project_link(project):
    return f"[{markdown(project['name'])}]({project['url']})"


def day(value):
    return value[:10] if value else "—"


def render_table(projects, columns):
    lines = ["| Project | " + " | ".join(columns) + " |", "|---|" + "---|" * len(columns)]
    for project in projects:
        values = []
        for column in columns:
            key = column.lower()
            value = project.get(key, project.get("updated_at") if key == "updated" else None)
            if key in ("updated", "published"):
                value = day(project.get(f"{key}_at"))
            elif key == "authors":
                value = ", ".join(value[:3]) + (" et al." if len(value) > 3 else "")
            elif key in ("stars", "downloads", "likes"):
                value = f"{value or 0:,}"
            values.append(markdown(value))
        lines.append("| " + project_link(project) + " | " + " | ".join(values) + " |")
    return lines


def activity_history(projects, commits, papers, changes):
    events = []
    update_labels = {
        "repository": "Repository metadata last updated",
        "agent": "Agent catalog entry last updated",
        "model": "Model repository last modified",
    }
    for project in projects:
        kind = project["kind"]
        if kind in update_labels and project.get("updated_at"):
            events.append({
                "timestamp": project["updated_at"], "type": kind,
                "label": project["name"], "detail": update_labels[kind], "url": project["url"],
            })
    for commit in commits:
        events.append({
            "timestamp": commit["committer_date"], "type": "commit",
            "label": commit["repo"], "detail": commit["message"] or "No commit message", "url": commit["url"],
        })
    for paper in papers:
        events.append({
            "timestamp": paper["published_at"], "type": "paper",
            "label": paper["title"], "detail": f"First arXiv submission · {paper['id']}",
            "url": f"https://arxiv.org/abs/{paper['id']}",
        })
        if paper["updated_at"] != paper["published_at"]:
            events.append({
                "timestamp": paper["updated_at"], "type": "paper",
                "label": paper["title"], "detail": f"Latest observed arXiv revision · {paper['id']}",
                "url": f"https://arxiv.org/abs/{paper['id']}",
            })
    for change in changes:
        if change["event"] == "changed":
            action = "Changed " + ", ".join(change.get("changes", {}))
        else:
            action = change["event"].capitalize()
        events.append({
            "timestamp": change["observed_at"], "type": "change",
            "label": change["name"], "detail": f"{action} · {change['kind']} · {change['id']}",
            "url": change["url"],
        })
    return sorted(
        events,
        key=lambda event: (event["timestamp"], event["type"], event["label"], event["detail"]),
        reverse=True,
    )


ACTIVITY_TYPES = ("repository", "agent", "model", "paper", "commit", "change")
ACTIVITY_COLORS = {
    "repository": "#4a86e8",
    "agent": "#d54f8a",
    "model": "#8864c7",
    "paper": "#b7791f",
    "commit": "#2e9e5b",
    "change": "#d64550",
}


def activity_daily(events, now, days=30):
    start = now.date() - timedelta(days=days - 1)
    counts = {
        start + timedelta(days=offset): {kind: 0 for kind in ACTIVITY_TYPES}
        for offset in range(days)
    }
    for event in events:
        timestamp = parse_time(event["timestamp"])
        kind = event["type"]
        if timestamp and start <= timestamp.date() <= now.date() and kind in ACTIVITY_TYPES:
            counts[timestamp.date()][kind] += 1
    return [
        {"date": day.isoformat(), "counts": counts[day]}
        for day in sorted(counts)
    ]


def render_activity_svg(daily):
    width, height = 960, 360
    left, right, top, bottom = 65, 25, 105, 285
    plot_width, plot_height = width - left - right, bottom - top
    maximum = max((count for row in daily for count in row["counts"].values()), default=0)
    y_max = max(1, maximum)
    x = lambda index: left + (index / (len(daily) - 1) if len(daily) > 1 else 0.5) * plot_width
    y = lambda value: bottom - value / y_max * plot_height
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="activity-title activity-desc">',
        '<title id="activity-title">Daily RWKV activity by category</title>',
        f'<desc id="activity-desc">Six line series for repositories, agents, models, papers, commits, and changes from {daily[0]["date"]} through {daily[-1]["date"]}.</desc>',
        '''<style>
text { font-family: system-ui, sans-serif; font-size: 12px; fill: #24292f; }
.grid { stroke: #d0d7de; stroke-width: 1; }
.axis { stroke: #8c959f; stroke-width: 1; }
@media (prefers-color-scheme: dark) { text { fill: #f0f6fc; } .grid { stroke: #30363d; } }
</style>''',
        '<text x="20" y="25" style="font-size:18px;font-weight:600">Activity over time</text>',
    ]
    for index, kind in enumerate(ACTIVITY_TYPES):
        column, row = index % 3, index // 3
        legend_x, legend_y = 70 + column * 285, 52 + row * 25
        total = sum(day["counts"][kind] for day in daily)
        parts.append(f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 22}" y2="{legend_y}" stroke="{ACTIVITY_COLORS[kind]}" stroke-width="3"/>')
        parts.append(f'<text x="{legend_x + 30}" y="{legend_y + 4}">{kind.capitalize()} ({total:,})</text>')
    for tick in range(5):
        value = y_max * tick / 4
        tick_y = y(value)
        parts.append(f'<line class="grid" x1="{left}" y1="{tick_y:.2f}" x2="{width - right}" y2="{tick_y:.2f}"/>')
        parts.append(f'<text x="{left - 10}" y="{tick_y + 4:.2f}" text-anchor="end">{value:.1f}</text>')
    parts.append(f'<line class="axis" x1="{left}" y1="{bottom}" x2="{width - right}" y2="{bottom}"/>')
    tick_indices = sorted({0, len(daily) // 4, len(daily) // 2, 3 * len(daily) // 4, len(daily) - 1})
    for index in tick_indices:
        anchor = "start" if index == 0 else "end" if index == len(daily) - 1 else "middle"
        parts.append(f'<text x="{x(index):.2f}" y="312" text-anchor="{anchor}">{daily[index]["date"]}</text>')
    for kind in ACTIVITY_TYPES:
        path = " ".join(
            f"{'M' if index == 0 else 'L'}{x(index):.2f},{y(day['counts'][kind]):.2f}"
            for index, day in enumerate(daily)
        )
        parts.append(f'<path data-type="{kind}" d="{path}" fill="none" stroke="{ACTIVITY_COLORS[kind]}" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>')
    parts.append('<text x="20" y="342" style="font-size:11px">Events per UTC day · trailing 30 days</text>')
    parts.append('</svg>')
    return "\n".join(parts) + "\n"


def render_activity_table(events, limit):
    lines = [
        "| Date (UTC) | Type | Activity | Details |",
        "|---|---|---|---|",
    ]
    for event in events[:limit]:
        timestamp = event["timestamp"][:16].replace("T", " ")
        activity = f"[{markdown(event['label'])}]({event['url']})"
        lines.append(f"| {timestamp} | {markdown(event['type'].capitalize())} | {activity} | {markdown(event['detail'])} |")
    return lines


def render_type_document(title, description, source, projects, columns, generated_at):
    return "\n".join([
        f"# {title}",
        "",
        description,
        "",
        f"**Records:** {len(projects):,} · **Snapshot:** {generated_at} · **[Source]({source})**",
        "",
        *render_table(projects, columns),
        "",
    ])


def render_type_documents(data):
    projects = data["projects"]
    repositories = [p for p in projects if p["kind"] == "repository"]
    agents = [p for p in projects if p["kind"] == "agent"]
    models = [p for p in projects if p["kind"] == "model"]
    papers = sorted(
        (p for p in projects if p["kind"] == "paper"),
        key=lambda p: (p["published_at"], p["id"]), reverse=True,
    )
    generated_at = data["generated_at"]
    return {
        Path("docs/github.md"): render_type_document(
            "GitHub Projects", "All non-fork GitHub repositories matching `rwkv`, ordered by repository update time.",
            GITHUB_SOURCE, repositories, ["Updated", "Stars", "Forks", "Language"], generated_at,
        ),
        Path("docs/models.md"): render_type_document(
            "Model Updates", "All Hugging Face models matching `rwkv`, ordered by last modification.",
            HF_SOURCE, models, ["Updated", "Downloads", "Likes"], generated_at,
        ),
        Path("docs/agents.md"): render_type_document(
            "Agent Projects", "All projects from the RWKV agent catalog, ordered by catalog update time.",
            AGENT_SOURCE, agents, ["Updated", "Stars", "Classification"], generated_at,
        ),
        Path("docs/papers.md"): render_type_document(
            "arXiv Papers", "All papers matching the RWKV title/abstract query, ordered by first arXiv submission.",
            ARXIV_SOURCE, papers, ["Published", "Updated", "Authors"], generated_at,
        ),
    }


def render_readme(data, history_state, activity):
    counts = {kind: sum(p["kind"] == kind for p in data["projects"]) for kind in ("repository", "agent", "model", "paper")}
    main = data["main_model"]
    freshness = "yes" if main["new_model_commit_within_31_days"] else "no"
    lines = [
        "# RWKV Project Tracker",
        "",
        "A daily summary of RWKV projects, model updates, agent projects, papers, and development activity.",
        "",
        f"**Last fetched:** {data['generated_at']} · **[Complete JSON data](data.json)**",
        "",
        "## Overview",
        "",
        "| Type | Records | Complete Markdown list |",
        "|---|---:|---|",
        f"| GitHub projects | {counts['repository']:,} | [github.md](docs/github.md) |",
        f"| Model updates | {counts['model']:,} | [models.md](docs/models.md) |",
        f"| Agent projects | {counts['agent']:,} | [agents.md](docs/agents.md) |",
        f"| arXiv papers | {counts['paper']:,} | [papers.md](docs/papers.md) |",
        "",
        "## Activity over time",
        "",
        "![Daily activity for repositories, agents, models, papers, commits, and changes](docs/activity.svg)",
        "",
        "Each line counts one activity category per UTC day over the trailing 30 days. [Daily series data](history/activity-daily.jsonl).",
        "",
        "## Latest activity",
        "",
        "The 25 most recent events across all sources. GitHub Pages shows the latest 100; the [activity JSONL](history/activity.jsonl) contains the complete derived feed.",
        "",
        *render_activity_table(activity, 25),
        "",
        "## Main model",
        "",
        f"Tracked model: [{MAIN_MODEL}]({main['url']}).",
        "",
        f"- Source repository last changed: **{day(main['source_last_modified'])}**.",
        f"- Latest observed `new model` commit: **{day(main['latest_new_model_commit_at'])}**.",
        f"- `new model` commit within 31 days: **{freshness}**.",
        f"- Newest date embedded in a current checkpoint filename: **{main['latest_current_checkpoint_filename_date'] or '—'}**.",
        "",
        "The 31-day indicator describes observed commit titles, not a guaranteed release schedule.",
        "",
        "## Activity history",
        "",
        "GitHub Pages includes auto-updating charts for project totals, daily commits, project changes, and arXiv first submissions.",
        "",
        "Raw history: [activity](history/activity.jsonl) · [daily counts](history/daily.jsonl) · [project changes](history/changes.jsonl) · [commits](history/commits.jsonl) · [papers](history/papers.jsonl).",
        "",
        f"Commit collection begins **{day(history_state['history_start'])}** and extends daily. Commits are observed on default branches. Project-count history begins with the first daily snapshot; earlier counts are never fabricated.",
        "",
        "Hugging Face results are matching model repositories, not independently verified weights. arXiv results mention RWKV in their title or abstract; publication dates mean first arXiv submission, not journal publication.",
        "",
    ]
    return "\n".join(lines)


def atomic_write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def main():
    now = datetime.now(UTC)
    snapshot_path = Path("data.json")
    previous = json.loads(snapshot_path.read_text(encoding="utf-8")) if snapshot_path.exists() else {"projects": [], "sources": {}}
    repositories = get_repositories(now)
    agents = get_agents()
    models = get_models()
    papers = get_papers()
    main_model = get_main_model(now)
    commits, history_state = commit_history(repositories, now)
    projects = sorted(
        repositories + agents + models + papers,
        key=lambda p: (p["updated_at"] or "", p["kind"], p["id"]),
        reverse=True,
    )
    data = {
        "schema_version": 1,
        "generated_at": now.isoformat().replace("+00:00", "Z"),
        "sources": {
            "repository": GITHUB_SOURCE, "agent": AGENT_SOURCE,
            "model": HF_SOURCE, "paper": ARXIV_SOURCE,
        },
        "main_model": main_model,
        "projects": projects,
    }
    changes = project_changes(previous, projects, now)
    daily = daily_history(projects, changes, parse_time(history_state["history_start"]), now)
    papers_observed = paper_history(papers, now)
    activity = activity_history(projects, commits, papers_observed, changes)
    activity_by_day = activity_daily(activity, now)
    outputs = {
        snapshot_path: json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        Path("README.md"): render_readme(data, history_state, activity),
        Path("docs/activity.svg"): render_activity_svg(activity_by_day),
        HISTORY_DIR / "activity.jsonl": jsonl(activity),
        HISTORY_DIR / "activity-daily.jsonl": jsonl(activity_by_day),
        HISTORY_DIR / "daily.jsonl": jsonl(daily),
        HISTORY_DIR / "changes.jsonl": jsonl(changes),
        HISTORY_DIR / "commits.jsonl": jsonl(commits),
        HISTORY_DIR / "papers.jsonl": jsonl(papers_observed),
        HISTORY_DIR / "state.json": json.dumps(history_state, indent=2) + "\n",
        **render_type_documents(data),
    }
    for path, content in outputs.items():
        atomic_write(path, content)
    print(
        f"Updated {len(repositories)} repositories, {len(agents)} agents, "
        f"{len(models)} models, {len(papers)} papers, and {len(commits)} historical commits."
    )


if __name__ == "__main__":
    main()
