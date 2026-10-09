"""
tmdb.py — Thin async wrapper around The Movie Database (TMDB) API.

Reads TMDB_API_KEY from the project-root .env file (or the real environment).
Accepts either the v3 "API Key" or the v4 "API Read Access Token".
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

# Project root = one level above this folder
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

TMDB_API_KEY = os.getenv("TMDB_API_KEY")
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"

MAX_ATTEMPTS = 3
RETRY_STATUSES = {429, 500, 502, 503, 504}
OVERVIEW_MAX_CHARS = 200

# httpx logs every request URL at INFO level. With a v3 key the URL contains
# ?api_key=..., so keep that logger quiet or the key ends up in your logs.
logging.getLogger("httpx").setLevel(logging.WARNING)

if not TMDB_API_KEY:
    # stderr only: in stdio mode, stdout is the MCP protocol channel.
    print(
        "WARNING: TMDB_API_KEY is not set; TMDB tools will return an error.",
        file=sys.stderr,
    )


class TMDBError(Exception):
    """Raised when a TMDB request fails (bad response, network error, missing key)."""



# HTTP layer
def _auth() -> tuple[dict[str, str], dict[str, str]]:
    """Return (headers, params) for the configured credential."""
    if not TMDB_API_KEY:
        raise TMDBError(
            "TMDB_API_KEY is not set. Add it to the .env file and restart the server."
        )
    if TMDB_API_KEY.startswith("eyJ"):
        # v4 read-access token (a JWT): send as a header, keeping it out of the URL.
        return {"Authorization": f"Bearer {TMDB_API_KEY}"}, {}
    return {}, {"api_key": TMDB_API_KEY}


def _error_message(resp: httpx.Response) -> str:
    try:
        detail = resp.json().get("status_message") or resp.text[:200]
    except ValueError:
        detail = resp.text[:200]

    if resp.status_code == 401:
        return "TMDB rejected the API key (401). Check TMDB_API_KEY."
    if resp.status_code == 404:
        return f"Not found on TMDB (404): {detail} Verify the id before retrying."
    return f"TMDB error {resp.status_code}: {detail}"


def _retry_delay(resp: httpx.Response, attempt: int) -> float:
    retry_after = resp.headers.get("Retry-After")
    if retry_after:
        try:
            return min(float(retry_after), 5.0)
        except ValueError:
            pass
    return 0.5 * 2 ** (attempt - 1)


async def _get(endpoint: str, params: dict[str, Any] | None = None) -> dict:
    headers, auth_params = _auth()
    query = {**(params or {}), **auth_params}

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            async with httpx.AsyncClient(timeout=15.0, headers=headers) as client:
                resp = await client.get(f"{TMDB_BASE_URL}{endpoint}", params=query)
        except httpx.HTTPError as e:
            if attempt == MAX_ATTEMPTS:
                # Type name only: the full message can contain the request URL.
                raise TMDBError(f"Could not reach TMDB ({type(e).__name__}).") from e
            await asyncio.sleep(0.5 * 2 ** (attempt - 1))
            continue

        if resp.status_code in RETRY_STATUSES and attempt < MAX_ATTEMPTS:
            await asyncio.sleep(_retry_delay(resp, attempt))
            continue

        if resp.status_code != 200:
            raise TMDBError(_error_message(resp))

        try:
            return resp.json()
        except ValueError as e:
            raise TMDBError("TMDB returned a response that is not valid JSON.") from e

    raise TMDBError("TMDB request failed.")  



# Normalizers
def _short(text: str | None, limit: int = OVERVIEW_MAX_CHARS) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _normalize(movie: dict) -> dict:
    """Compact movie dict for list results (cheap for the agent to read)."""
    return {
        "tmdb_id": movie.get("id"),
        "title": movie.get("title") or movie.get("name"),
        "release_date": movie.get("release_date") or None,
        "rating": round(movie.get("vote_average") or 0.0, 2),
        "vote_count": movie.get("vote_count") or 0,
        "overview": _short(movie.get("overview")),
    }


def _normalize_full(movie: dict) -> dict:
    """Full movie dict (untruncated overview, poster, popularity) for detail views."""
    poster = movie.get("poster_path")
    result = _normalize(movie)
    result.update(
        {
            "overview": movie.get("overview") or "",
            "popularity": round(movie.get("popularity") or 0.0, 2),
            "poster_url": f"{TMDB_IMAGE_BASE}{poster}" if poster else None,
        }
    )
    return result


# Public functions
async def search_movies(query: str, page: int = 1) -> list[dict]:
    data = await _get(
        "/search/movie",
        {"query": query, "page": page, "include_adult": False},
    )
    return [_normalize(m) for m in data.get("results", [])]


async def get_movie_details(tmdb_id: int) -> dict:
    data = await _get(
        f"/movie/{tmdb_id}",
        {"append_to_response": "credits,keywords"},
    )
    result = _normalize_full(data)
    result.update(
        {
            "runtime": data.get("runtime"),
            "tagline": data.get("tagline") or "",
            "status": data.get("status"),
            "genres": [g["name"] for g in data.get("genres", [])],
            "cast": [
                c["name"]
                for c in data.get("credits", {}).get("cast", [])[:10]
            ],
            "director": next(
                (
                    c["name"]
                    for c in data.get("credits", {}).get("crew", [])
                    if c.get("job") == "Director"
                ),
                None,
            ),
            "keywords": [
                k["name"]
                for k in data.get("keywords", {}).get("keywords", [])[:15]
            ],
        }
    )
    return result


async def get_similar_movies(tmdb_id: int, page: int = 1) -> list[dict]:
    data = await _get(f"/movie/{tmdb_id}/similar", {"page": page})
    return [_normalize(m) for m in data.get("results", [])]


async def get_recommended_movies(tmdb_id: int, page: int = 1) -> list[dict]:
    data = await _get(f"/movie/{tmdb_id}/recommendations", {"page": page})
    return [_normalize(m) for m in data.get("results", [])]


async def get_trending_movies(window: str = "week", page: int = 1) -> list[dict]:
    if window not in ("day", "week"):
        window = "week"
    data = await _get(f"/trending/movie/{window}", {"page": page})
    return [_normalize(m) for m in data.get("results", [])]


async def discover_movies(
    genre_id: int | None = None,
    year: int | None = None,
    min_rating: float | None = None,
    sort_by: str = "popularity.desc",
    page: int = 1,
) -> list[dict]:
    params: dict[str, Any] = {"sort_by": sort_by, "page": page}
    if genre_id is not None:
        params["with_genres"] = genre_id
    if year is not None:
        params["primary_release_year"] = year
    if min_rating is not None:
        params["vote_average.gte"] = min_rating
        params["vote_count.gte"] = 100

    data = await _get("/discover/movie", params)
    return [_normalize(m) for m in data.get("results", [])]


# Genres (static list, fetched once per server run)
_genres_cache: list[dict] | None = None

_GENRE_ALIASES = {
    "sci-fi": "science fiction",
    "scifi": "science fiction",
    "sci fi": "science fiction",
    "animated": "animation",
    "doc": "documentary",
}


async def get_genres() -> list[dict]:
    global _genres_cache
    if _genres_cache is None:
        data = await _get("/genre/movie/list")
        _genres_cache = data.get("genres", [])
    return list(_genres_cache)


async def resolve_genre_id(name: str) -> int | None:
    """Map a genre name like 'sci-fi' or 'Science Fiction' to a TMDB genre id."""
    wanted = name.strip().lower()
    wanted = _GENRE_ALIASES.get(wanted, wanted)
    for genre in await get_genres():
        if genre["name"].lower() == wanted:
            return genre["id"]
    return None