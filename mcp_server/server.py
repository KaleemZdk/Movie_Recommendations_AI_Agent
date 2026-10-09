"""
mcp/server.py — MCP server exposing movie-recommendation tools.

Combines:
  * TMDB API (search / details / similar / trending / discover)
  * Local SQLite state (favorites / watching / reviews)

Run (stdio, for LangGraph):
    python -m mcp.server
    python mcp/server.py

Run (HTTP transport):
    python mcp/server.py --http
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP
try:
    from . import database, tmdb  
except ImportError:  
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import database  
    import tmdb      



# DB path fix: database.py defaults to mcp/movies.db.
# If the project has a data/ directory, use data/movies.db instead.

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATA_DB = _PROJECT_ROOT / "data" / "movies.db"
if _DATA_DB.parent.exists():
    database.DB_PATH = _DATA_DB

database.init_db()

# FastMCP server
mcp = FastMCP(
    "movie-recommender",
    instructions=(
        "Tools for an AI movie-recommendation agent.\n"
        "• Search & discover movies via TMDB (search_movies, get_movie_details, "
        "get_similar_movies, get_recommended_movies, get_trending_movies, "
        "discover_movies, list_genres).\n"
        "• Manage user state in a local SQLite DB (add_favorite, list_favorites, "
        "remove_favorite, set_watching, list_watching, remove_watching, "
        "save_review, get_review, list_reviews).\n"
        "• get_user_profile returns everything the user has saved.\n"
        "• recommend_for_user builds a personalized list from their reviews."
    ),
)

# Helpers
def _dumps(payload: Any) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False, default=str)


def _err(message: str, **extra: Any) -> str:
    return _dumps({"error": message, **extra})


# TMDB TOOLS
@mcp.tool()
async def search_movies(query: str, page: int = 1) -> str:
    """Search TMDB for movies matching a title or keyword.

    Use this first to resolve a user's free-text request into a tmdb_id that
    other tools accept.

    Args:
        query: Free-text query, e.g. "Blade Runner", "Nolan sci-fi".
        page: 1-based page number (20 results per page).
    """
    try:
        results = await tmdb.search_movies(query, page)
    except tmdb.TMDBError as e:
        return _err(str(e))

    return _dumps(
        {
            "query": query,
            "page": page,
            "count": len(results),
            "results": results,
        }
    )


@mcp.tool()
async def get_movie_details(tmdb_id: int) -> str:
    """Get full details for one movie: cast, director, runtime, keywords, etc.

    Args:
        tmdb_id: TMDB numeric id (returned by `search_movies`).
    """
    try:
        details = await tmdb.get_movie_details(tmdb_id)
    except tmdb.TMDBError as e:
        return _err(str(e), tmdb_id=tmdb_id)

    return _dumps(details)


@mcp.tool()
async def get_similar_movies(tmdb_id: int, page: int = 1) -> str:
    """Return TMDB's 'similar movies' for a given movie.

    Good for "more like this" follow-ups.

    Args:
        tmdb_id: Reference movie id.
        page: 1-based page number.
    """
    try:
        results = await tmdb.get_similar_movies(tmdb_id, page)
    except tmdb.TMDBError as e:
        return _err(str(e), tmdb_id=tmdb_id)

    return _dumps({"tmdb_id": tmdb_id, "count": len(results), "results": results})


@mcp.tool()
async def get_recommended_movies(tmdb_id: int, page: int = 1) -> str:
    """Return TMDB's 'recommended for you' list for a given movie.

    Usually more personalized than `get_similar_movies`.

    Args:
        tmdb_id: Reference movie id.
        page: 1-based page number.
    """
    try:
        results = await tmdb.get_recommended_movies(tmdb_id, page)
    except tmdb.TMDBError as e:
        return _err(str(e), tmdb_id=tmdb_id)

    return _dumps({"tmdb_id": tmdb_id, "count": len(results), "results": results})


@mcp.tool()
async def get_trending_movies(window: str = "week", page: int = 1) -> str:
    """Get currently trending movies on TMDB.

    Args:
        window: "day" or "week" (default "week").
        page: 1-based page number.
    """
    try:
        results = await tmdb.get_trending_movies(window, page)
    except tmdb.TMDBError as e:
        return _err(str(e))

    return _dumps({"window": window, "count": len(results), "results": results})


@mcp.tool()
async def discover_movies(
    genre_id: int | None = None,
    year: int | None = None,
    min_rating: float | None = None,
    sort_by: str = "popularity.desc",
    page: int = 1,
) -> str:
    """Discover movies using TMDB filters (genre, year, rating, sorting).

    Use `list_genres` first if you need a genre name → id mapping.

    Args:
        genre_id: TMDB genre id (e.g. 878 = Sci-Fi).
        year: Primary release year (inclusive).
        min_rating: Minimum vote average (0-10). Only titles with 100+ votes.
        sort_by: TMDB sort key, e.g. "popularity.desc", "vote_average.desc",
                 "primary_release_date.desc".
        page: 1-based page number.
    """
    try:
        results = await tmdb.discover_movies(
            genre_id=genre_id,
            year=year,
            min_rating=min_rating,
            sort_by=sort_by,
            page=page,
        )
    except tmdb.TMDBError as e:
        return _err(str(e))

    return _dumps(
        {
            "filters": {
                "genre_id": genre_id,
                "year": year,
                "min_rating": min_rating,
                "sort_by": sort_by,
            },
            "count": len(results),
            "results": results,
        }
    )


@mcp.tool()
async def list_genres() -> str:
    """List all TMDB movie genres as {id, name} pairs.

    Call this before `discover_movies` when the user names a genre by name.
    """
    try:
        genres = await tmdb.get_genres()
    except tmdb.TMDBError as e:
        return _err(str(e))

    return _dumps({"genres": genres})


# =========================================================================== #
# LOCAL DB — FAVORITES
# =========================================================================== #

@mcp.tool()
def add_favorite(tmdb_id: int, title: str) -> str:
    """Save a movie to the user's favorites list.

    Args:
        tmdb_id: TMDB movie id.
        title: Movie title (for display in the local DB).
    """
    added = database.add_favorite(tmdb_id, title)
    return _dumps(
        {
            "success": added,
            "already_existed": not added,
            "tmdb_id": tmdb_id,
            "title": title,
        }
    )


@mcp.tool()
def list_favorites() -> str:
    """List every movie the user has favorited."""
    favs = database.list_favorites()
    return _dumps({"count": len(favs), "favorites": favs})


@mcp.tool()
def remove_favorite(tmdb_id: int) -> str:
    """Remove a movie from the user's favorites list.

    Args:
        tmdb_id: TMDB movie id.
    """
    removed = database.remove_favorite(tmdb_id)
    return _dumps({"success": removed, "tmdb_id": tmdb_id})


# =========================================================================== #
# LOCAL DB — WATCHING
# =========================================================================== #

@mcp.tool()
def set_watching(tmdb_id: int, title: str) -> str:
    """Mark a movie as 'currently watching'.

    Args:
        tmdb_id: TMDB movie id.
        title: Movie title.
    """
    added = database.set_watching(tmdb_id, title)
    return _dumps(
        {
            "success": added,
            "already_existed": not added,
            "tmdb_id": tmdb_id,
            "title": title,
        }
    )


@mcp.tool()
def list_watching() -> str:
    """List all movies the user is currently watching."""
    items = database.list_watching()
    return _dumps({"count": len(items), "watching": items})


@mcp.tool()
def remove_watching(tmdb_id: int) -> str:
    """Remove a movie from the user's watching list.

    Args:
        tmdb_id: TMDB movie id.
    """
    removed = database.remove_watching(tmdb_id)
    return _dumps({"success": removed, "tmdb_id": tmdb_id})


# =========================================================================== #
# LOCAL DB — REVIEWS
# =========================================================================== #

@mcp.tool()
def save_review(tmdb_id: int, rating: float, comment: str = "") -> str:
    """Save or update a review for a movie (rating must be 0–10).

    Args:
        tmdb_id: TMDB movie id.
        rating: Numeric rating between 0 and 10.
        comment: Optional free-text review.
    """
    try:
        database.save_review(tmdb_id, rating, comment)
    except ValueError as e:
        return _err(str(e), tmdb_id=tmdb_id)

    return _dumps(
        {"success": True, "tmdb_id": tmdb_id, "rating": rating}
    )


@mcp.tool()
def get_review(tmdb_id: int) -> str:
    """Get the user's review for a single movie (or null if none).

    Args:
        tmdb_id: TMDB movie id.
    """
    review = database.get_review(tmdb_id)
    return _dumps({"tmdb_id": tmdb_id, "review": review})


@mcp.tool()
def list_reviews() -> str:
    """List all reviews the user has written."""
    reviews = database.list_reviews()
    return _dumps({"count": len(reviews), "reviews": reviews})


# =========================================================================== #
# AGGREGATE / PERSONALIZATION
# =========================================================================== #

@mcp.tool()
def get_user_profile() -> str:
    """Return the user's full saved state: favorites, watching, and reviews.

    Call this whenever the agent needs context to personalize recommendations.
    """
    return _dumps(
        {
            "favorites": database.list_favorites(),
            "watching": database.list_watching(),
            "reviews": database.list_reviews(),
        }
    )


@mcp.tool()
async def recommend_for_user(limit: int = 10) -> str:
    """Build a personalized recommendation list from the user's reviews.

    Strategy:
      1. Use the user's top-rated reviews as seed movies.
      2. Pull TMDB recommendations for each seed.
      3. Rank candidates by (seed weight + TMDB rating) and dedupe.
      4. Exclude anything already saved in favorites / watching / reviews.

    Falls back to trending movies if the user has no data yet.

    Args:
        limit: Maximum number of recommendations to return (1-20).
    """
    limit = max(1, min(limit, 20))

    reviews = database.list_reviews()
    favorites = database.list_favorites()
    watching = database.list_watching()

    seen_ids = {r["tmdb_id"] for r in reviews}
    seen_ids |= {f["tmdb_id"] for f in favorites}
    seen_ids |= {w["tmdb_id"] for w in watching}

    # ---- Fall back to trending if we have no signal ----
    if not reviews:
        try:
            results = await tmdb.get_trending_movies("week")
        except tmdb.TMDBError as e:
            return _err(str(e))

        return _dumps(
            {
                "strategy": "trending (no user reviews yet)",
                "count": min(len(results), limit),
                "results": results[:limit],
            }
        )

    # ---- Seeds: top 3 reviews by rating ----
    seeds = sorted(reviews, key=lambda r: -r["rating"])[:3]

    # ---- Gather candidates from each seed ----
    candidates: dict[int, dict] = {}

    for seed in seeds:
        weight = seed["rating"] / 10.0
        try:
            recs = await tmdb.get_recommended_movies(seed["tmdb_id"])
        except tmdb.TMDBError:
            continue

        for rec in recs:
            rid = rec["tmdb_id"]
            if not rid or rid in seen_ids:
                continue

            if rid not in candidates:
                candidates[rid] = {"movie": rec, "score": 0.0, "based_on": []}

            candidates[rid]["score"] += weight + (rec.get("rating") or 0) / 10.0
            candidates[rid]["based_on"].append(seed["tmdb_id"])

    ranked = sorted(candidates.values(), key=lambda c: -c["score"])

    results = [
        {
            **c["movie"],
            "score": round(c["score"], 2),
            "based_on": c["based_on"],
        }
        for c in ranked[:limit]
    ]

    return _dumps(
        {
            "strategy": "personalized from reviews",
            "seeds": [
                {"tmdb_id": s["tmdb_id"], "rating": s["rating"]} for s in seeds
            ],
            "count": len(results),
            "results": results,
        }
    )


# =========================================================================== #
# RESOURCES
# =========================================================================== #

@mcp.resource("movies://user-profile")
def user_profile_resource() -> str:
    """The user's saved state (favorites / watching / reviews) as JSON."""
    return _dumps(
        {
            "favorites": database.list_favorites(),
            "watching": database.list_watching(),
            "reviews": database.list_reviews(),
        }
    )


@mcp.resource("movies://genres")
async def genres_resource() -> str:
    """TMDB movie genres as JSON."""
    try:
        genres = await tmdb.get_genres()
    except tmdb.TMDBError as e:
        return _dumps({"error": str(e)})
    return _dumps({"genres": genres})


# =========================================================================== #
# PROMPT
# =========================================================================== #

@mcp.prompt()
def recommend_prompt(taste: str) -> str:
    """Reusable prompt template for a recommendation turn."""
    return (
        f"The user describes their taste as: {taste}\n\n"
        "Workflow:\n"
        "1. Call `get_user_profile` to see what they already like.\n"
        "2. Call `search_movies` to resolve any titles they mention to tmdb_ids.\n"
        "3. Call `recommend_for_user` for a personalized baseline.\n"
        "4. Enrich the top picks with `get_movie_details`.\n\n"
        "Return at most 5 movies. For each: title, year, TMDB rating, "
        "a one-sentence spoiler-free pitch, and why it fits their taste."
    )


# =========================================================================== #
# ENTRYPOINT
# =========================================================================== #

if __name__ == "__main__":
    if "--http" in sys.argv:
        mcp.settings.host = "127.0.0.1"
        mcp.settings.port = 8000
        mcp.run(transport="streamable-http")
    else:
        mcp.run(transport="stdio")