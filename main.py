from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from analysis import analyze
from scraper import fetch_game_relay, fetch_games, fetch_hitter_advanced_rankings, fetch_player_matchup, fetch_player_profile, fetch_players, fetch_standings, fetch_team_season_stats

app = FastAPI(title="BASELAB API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/players")
def players(
    season: int = Query(default=datetime.now().year, ge=1982, le=2100),
    position: str = Query(default="hitter", pattern="^(hitter|pitcher)$"),
) -> dict:
    try:
        records = fetch_players(season, position)
        if position == "hitter":
            team_games = max(int(record.get("G", 0) or 0) for record in records)
        else:
            hitter_records = fetch_players(season, "hitter")
            team_games = max(int(record.get("G", 0) or 0) for record in hitter_records)

        if position == "hitter":
            required = team_games * 3.1
            qualified = [record for record in records if float(record.get("PA", 0) or 0) >= required]
            advanced = fetch_hitter_advanced_rankings(season)
            for record in qualified:
                record.update(advanced.get(f"{record.get('name')}|{record.get('team')}", {}))
            qualification_label = f"규정타석 {required:.1f} 이상 ({team_games}경기 × 3.1)"
        else:
            required = float(team_games)
            qualified = [record for record in records if float(record.get("IP", 0) or 0) >= required]
            qualification_label = f"규정이닝 {required:.0f}이닝 이상 ({team_games}경기)"
        return {
            "season": season,
            "position": position,
            "team_games": team_games,
            "qualification": required,
            "qualification_label": qualification_label,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "players": analyze(qualified, position),
        }
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"KBO 데이터 수집 실패: {exc}") from exc


@app.get("/api/games")
def games(date: str = Query(default_factory=lambda: datetime.now().strftime("%Y%m%d"), pattern=r"^\d{8}$")) -> dict:
    try:
        return {"date": date, "games": fetch_games(date), "fetched_at": datetime.now(timezone.utc).isoformat()}
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"KBO 경기 데이터 수집 실패: {exc}") from exc


@app.get("/api/game-relay")
def game_relay(
    game_id: str = Query(min_length=10),
    inning: int | None = Query(default=None, ge=1, le=15),
) -> dict:
    try:
        return fetch_game_relay(game_id.strip(), inning)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"경기 라인업 수집 실패: {exc}") from exc


@app.get("/api/standings")
def standings(season: int = Query(default=datetime.now().year, ge=1982, le=2100)) -> dict:
    try:
        rows = fetch_standings(season)
        team_stats = fetch_team_season_stats(season)
        for row in rows:
            row.update(team_stats.get(row["team"], {}))
        return {
            "season": season,
            "standings": rows,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"KBO 팀 순위 수집 실패: {exc}") from exc


@app.get("/api/matchup")
def matchup(pitcher: str = Query(min_length=1), batter: str = Query(min_length=1)) -> dict:
    try:
        return fetch_player_matchup(pitcher.strip(), batter.strip())
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"투수·타자 상대전적 수집 실패: {exc}") from exc


@app.get("/api/player-profile")
def player_profile(
    name: str = Query(min_length=1),
    team: str = Query(default=""),
    position: str = Query(default="hitter", pattern="^(hitter|pitcher)$"),
) -> dict:
    try:
        return fetch_player_profile(name.strip(), team.strip(), position)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"선수 상세 기록 수집 실패: {exc}") from exc
