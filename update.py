#!/usr/bin/env python3
"""Refresh the RWKV tracker using only the Python standard library."""

import html
import json
import os
import re
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen

GITHUB_SOURCE = "https://github.com/search?o=desc&p=1&q=rwkv&s=updated&type=Repositories"
AGENT_SOURCE = "https://agent.objects.rwkvos.com/en"
AGENT_FEED = "https://agent.objects.rwkvos.com/agent-data.json"
HF_SOURCE = "https://huggingface.co/models?sort=modified&search=rwkv"
MAIN_MODEL = "BlinkDL/rwkv7-g1"
USER_AGENT = "rwkv-projects-tracking/1.0"


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


def markdown(value):
    text = html.escape(str(value if value is not None else "—"))
    return text.replace("|", "\\|").replace("\r", " ").replace("\n", " ")


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
            if key == "updated":
                value = day(project["updated_at"])
            elif key in ("stars", "downloads", "likes"):
                value = f"{value or 0:,}"
            values.append(markdown(value))
        lines.append("| " + project_link(project) + " | " + " | ".join(values) + " |")
    return lines


def render_readme(data):
    repositories = [p for p in data["projects"] if p["kind"] == "repository"]
    agents = [p for p in data["projects"] if p["kind"] == "agent"]
    models = [p for p in data["projects"] if p["kind"] == "model"]
    main = data["main_model"]
    freshness = "yes" if main["new_model_commit_within_31_days"] else "no"
    lines = [
        "# RWKV Project Tracker",
        "",
        "A dependency-free, daily snapshot of RWKV repositories, agent projects, and Hugging Face models.",
        "",
        f"**Last fetched:** {data['generated_at']} · **[Complete JSON data](data.json)**",
        "",
        "| Source | Projects |",
        "|---|---:|",
        f"| [GitHub repositories]({GITHUB_SOURCE}) | {len(repositories):,} |",
        f"| [Agent projects]({AGENT_SOURCE}) | {len(agents):,} |",
        f"| [Hugging Face models]({HF_SOURCE}) | {len(models):,} |",
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
        "## Recently updated",
        "",
        "The JSON dataset and GitHub Pages site contain every record; these tables show the 25 most recently updated projects per source.",
        "",
        "### GitHub repositories",
        "",
        *render_table(repositories[:25], ["Updated", "Stars", "Language"]),
        "",
        "### Agent projects",
        "",
        *render_table(agents[:25], ["Updated", "Stars", "Classification"]),
        "",
        "### Hugging Face models",
        "",
        *render_table(models[:25], ["Updated", "Downloads", "Likes"]),
        "",
        "## Update",
        "",
        "Run `python3 update.py`. Python 3.11+ is the only requirement. Set `GITHUB_TOKEN` to raise the GitHub search rate limit; the token is sent only to `api.github.com`.",
        "",
        "The scheduled workflow runs once daily, commits `data.json` and this README, and deploys `index.html` to GitHub Pages. A failed fetch leaves the previous snapshot untouched.",
        "",
        "GitHub repository search excludes forks by default. Creation-date shards work around its 1,000-result-per-query cap. Hugging Face results are matching model repositories; their contents are not independently verified as usable weights.",
        "",
    ]
    return "\n".join(lines)


def atomic_write(path, content):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def main():
    now = datetime.now(UTC)
    repositories = get_repositories(now)
    agents = get_agents()
    models = get_models()
    data = {
        "schema_version": 1,
        "generated_at": now.isoformat().replace("+00:00", "Z"),
        "sources": {"repository": GITHUB_SOURCE, "agent": AGENT_SOURCE, "model": HF_SOURCE},
        "main_model": get_main_model(now),
        "projects": sorted(
            repositories + agents + models,
            key=lambda p: (p["updated_at"] or "", p["kind"], p["id"]),
            reverse=True,
        ),
    }
    content = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    atomic_write(Path("data.json"), content)
    atomic_write(Path("README.md"), render_readme(data))
    print(f"Updated {len(repositories)} repositories, {len(agents)} agents, and {len(models)} models.")


if __name__ == "__main__":
    main()
