"""KBO 기록실 HTML 수집기. 영구 저장 없이 10분 메모리 캐시만 사용한다."""
from __future__ import annotations

import time
from typing import Any

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.koreabaseball.com/Record/Player"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; BaseLab/1.0; +local-analysis)",
    "Referer": "https://www.koreabaseball.com/Record/Player/HitterBasic/Basic1.aspx",
}
CACHE_TTL = 600
_cache: dict[str, tuple[float, list[dict[str, Any]]]] = {}
ARCHIVE_URL = "https://huggingface.co/datasets/juhonov/KBOresearch/resolve/main/kbo_{kind}_stats_2000_2025.json"
HISTORY_RANKING_URL = "https://www.yagoonara.com/api/rankings"
STANDINGS_URL = "https://www.yagoonara.com/api/standings"
PLAYER_SEARCH_URL = "https://www.yagoonara.com/api/players"
PLAYER_MATCHUP_URL = "https://www.yagoonara.com/api/matchups/player"

HITTER_COLUMNS = ["rank", "name", "team", "AVG", "G", "PA", "AB", "H", "2B", "3B", "HR", "RBI", "SB", "CS", "BB", "HBP", "SO", "GDP", "E"]
PITCHER_COLUMNS = ["rank", "name", "team", "ERA", "G", "W", "L", "SV", "HLD", "WPCT", "IP", "H", "HR", "BB", "HBP", "SO", "R", "ER", "WHIP"]


def _number(value: str) -> int | float | str:
    cleaned = value.strip().replace(",", "")
    if not cleaned or cleaned == "-":
        return 0
    if "/" in cleaned:  # 투구 이닝의 1/3, 2/3 보존
        whole, fraction = (cleaned.split() if " " in cleaned else ("0", cleaned))
        return round(float(whole) + ({"1/3": 1 / 3, "2/3": 2 / 3}.get(fraction, 0)), 2)
    try:
        return float(cleaned) if "." in cleaned else int(cleaned)
    except ValueError:
        return cleaned


def fetch_players(season: int, position: str) -> list[dict[str, Any]]:
    """KBO 공식 기록표 전체 페이지를 순회해 표준 dict 목록으로 반환한다."""
    key = f"{season}:{position}"
    if key in _cache and time.time() - _cache[key][0] < CACHE_TTL:
        return _cache[key][1]

    # KBO의 ASP.NET 기록표는 과거 시즌 변경에 브라우저 실행이 필요하다.
    # 2025년 이전은 KBO 원본을 연도별로 수집한 공개 아카이브를 메모리에서 필터링한다.
    if season < 2000:
        return _fetch_legacy_players(season, position)
    if season <= 2025:
        return _fetch_archived_players(season, position)

    is_hitter = position == "hitter"
    path = "HitterBasic/BasicOld.aspx" if is_hitter else "PitcherBasic/Basic1.aspx"
    columns = HITTER_COLUMNS if is_hitter else PITCHER_COLUMNS
    players: list[dict[str, Any]] = []

    # KBO 표는 qs_page에 따라 페이지가 바뀐다. 빈 페이지를 만나면 종료한다.
    for page in range(1, 8):
        response = requests.get(
            f"{BASE_URL}/{path}",
            params={"seasonId": season, "seriesId": "0", "page": page},
            headers=HEADERS,
            timeout=12,
        )
        response.raise_for_status()
        response.encoding = response.apparent_encoding or "utf-8"
        soup = BeautifulSoup(response.text, "html.parser")
        rows = soup.select("table.tData01 tbody tr")
        if not rows:
            rows = soup.select("table tbody tr")

        page_players = []
        for row in rows:
            values = [cell.get_text(" ", strip=True) for cell in row.select("th, td")]
            if len(values) < min(8, len(columns)) or not values[0].replace(".", "").isdigit():
                continue
            item = {column: _number(value) for column, value in zip(columns, values)}
            if not isinstance(item.get("name"), str):
                continue
            item["position"] = "타자" if is_hitter else "투수"
            page_players.append(item)
        if not page_players:
            break
        known = {(p["name"], p["team"]) for p in players}
        players.extend(p for p in page_players if (p["name"], p["team"]) not in known)
        if len(page_players) < 20:
            break

    if not players:
        raise RuntimeError("KBO 기록표를 찾지 못했습니다. 원본 페이지 구조를 확인해 주세요.")
    _cache[key] = (time.time(), players)
    return players


def _fetch_archived_players(season: int, position: str) -> list[dict[str, Any]]:
    archive_key = f"archive:{position}"
    if archive_key in _cache and time.time() - _cache[archive_key][0] < 86400:
        all_records = _cache[archive_key][1]
    else:
        kind = "hitter" if position == "hitter" else "pitcher"
        response = requests.get(ARCHIVE_URL.format(kind=kind), headers=HEADERS, timeout=30)
        response.raise_for_status()
        all_records = response.json().get("data", [])
        _cache[archive_key] = (time.time(), all_records)

    converted_by_player: dict[tuple[str, str], dict[str, Any]] = {}
    for raw in all_records:
        if int(raw.get("year", 0)) != season:
            continue
        item = {
            "rank": _number(str(raw.get("순위", 0))),
            "name": raw.get("선수명", ""),
            "team": raw.get("팀명", raw.get("teamName", "")),
            "position": "타자" if position == "hitter" else "투수",
        }
        fields = HITTER_COLUMNS[3:] if position == "hitter" else PITCHER_COLUMNS[3:]
        for field in fields:
            item[field] = _number(str(raw.get(field, 0)))
        # 극소 표본 선수가 타율/ERA 1위로 노출되지 않도록 기본 표본을 적용한다.
        if position == "hitter" and float(item.get("PA", 0) or 0) < 100:
            continue
        if position == "pitcher":
            innings = float(item.get("IP", 0) or 0)
            games = float(item.get("G", 0) or 0)
            if innings < 70 or not games or innings / games < 3:
                continue
        player_key = (str(item["name"]), str(item["team"]))
        sample_key = "PA" if position == "hitter" else "IP"
        previous = converted_by_player.get(player_key)
        if previous is None or float(item.get(sample_key, 0) or 0) > float(previous.get(sample_key, 0) or 0):
            converted_by_player[player_key] = item

    converted = list(converted_by_player.values())

    if not converted:
        raise RuntimeError(f"{season} 시즌 아카이브 기록을 찾지 못했습니다.")
    _cache[f"{season}:{position}"] = (time.time(), converted)
    return converted


def _fetch_legacy_players(season: int, position: str) -> list[dict[str, Any]]:
    """1982~1999 시즌의 규정 기록 순위를 공개 역사 기록 API에서 조합한다."""
    if not 1982 <= season <= 1999:
        raise RuntimeError(f"{season} 시즌은 KBO 정규시즌 범위가 아닙니다.")

    is_hitter = position == "hitter"
    stat_map = (
        {"AVG": "avg", "H": "hits", "HR": "hr", "RBI": "rbi", "OPS": "ops"}
        if is_hitter
        else {"ERA": "era", "IP": "innings", "W": "wins", "SO": "so", "WHIP": "whip"}
    )
    rankings: dict[str, list[dict[str, Any]]] = {}
    for field, stat in stat_map.items():
        response = requests.get(
            HISTORY_RANKING_URL,
            params={"type": position, "stat": stat, "period": season, "limit": 300},
            headers=HEADERS,
            timeout=20,
        )
        response.raise_for_status()
        rankings[field] = response.json().get("data", [])

    primary_field = "AVG" if is_hitter else "ERA"
    supplements = {
        field: {int(row["player_id"]): _number(str(row.get("value", 0))) for row in rows}
        for field, rows in rankings.items()
    }
    players = []
    for row in rankings[primary_field]:
        player_id = int(row["player_id"])
        item: dict[str, Any] = {
            "rank": int(row.get("rank") or 0),
            "name": row.get("player_name", ""),
            "team": row.get("team_name", ""),
            "position": "타자" if is_hitter else "투수",
            "G": int(row.get("games") or 0),
        }
        if is_hitter:
            item["PA"] = int(row.get("pa") or 0)
        else:
            item["IP"] = _number(str(row.get("innings") or supplements["IP"].get(player_id, 0)))
        for field in stat_map:
            item[field] = supplements[field].get(player_id, 0)
        players.append(item)

    if not players:
        raise RuntimeError(f"{season} 시즌 역사 기록을 찾지 못했습니다.")
    _cache[f"{season}:{position}"] = (time.time(), players)
    return players


def fetch_games(game_date: str) -> list[dict[str, Any]]:
    """KBO 게임센터 API에서 일정·결과·예고 선발을 가져온다."""
    response = requests.post(
        "https://www.koreabaseball.com/ws/Main.asmx/GetKboGameList",
        data={"leId": "1", "srId": "0,1,3,4,5,6,7,8,9", "date": game_date},
        headers={**HEADERS, "Referer": "https://www.koreabaseball.com/Schedule/GameCenter/Main.aspx"},
        timeout=12,
    )
    response.raise_for_status()
    games = []
    for game in response.json().get("game", []):
        is_final = bool(game.get("GAME_RESULT_CK"))
        is_cancelled = str(game.get("CANCEL_SC_ID", "0")) != "0"
        game_state = str(game.get("GAME_STATE_SC") or "1")
        if is_cancelled:
            status = game.get("CANCEL_SC_NM") or "취소"
        elif is_final or game_state == "3":
            status = "종료"
        elif game_state == "2":
            status = "경기중"
        else:
            status = "예정"
        games.append({
            "game_id": game.get("G_ID"),
            "date": game.get("G_DT"),
            "time": game.get("G_TM"),
            "stadium": game.get("S_NM"),
            "away": game.get("AWAY_NM"),
            "home": game.get("HOME_NM"),
            "away_score": int(game.get("T_SCORE_CN") or 0),
            "home_score": int(game.get("B_SCORE_CN") or 0),
            "away_starter": (game.get("T_PIT_P_NM") or "미정").strip(),
            "home_starter": (game.get("B_PIT_P_NM") or "미정").strip(),
            "status": status,
            "broadcast": game.get("TV_IF") or "",
            "inning": int(game.get("GAME_INN_NO") or 0),
            "inning_half": game.get("GAME_TB_SC_NM") or "",
            "balls": min(int(game.get("BALL_CN") or 0), 3),
            "strikes": min(int(game.get("STRIKE_CN") or 0), 2),
            "outs": min(int(game.get("OUT_CN") or 0), 2),
            "first_base": bool(int(game.get("B1_BAT_ORDER_NO") or 0)),
            "second_base": bool(int(game.get("B2_BAT_ORDER_NO") or 0)),
            "third_base": bool(int(game.get("B3_BAT_ORDER_NO") or 0)),
            "current_away_player": (game.get("T_P_NM") or "").strip(),
            "current_home_player": (game.get("B_P_NM") or "").strip(),
        })
    return games


def fetch_game_relay(game_id: str, inning: int | None = None) -> dict[str, Any]:
    """네이버 경기 중계 공개 응답에서 실제 라인업과 회차별 타자를 읽는다."""
    naver_game_id = game_id if len(game_id) > 14 else f"{game_id}{game_id[:4]}"
    params = {"inning": inning} if inning else None
    response = requests.get(
        f"https://api-gw.sports.naver.com/schedule/games/{naver_game_id}/relay",
        params=params,
        headers={**HEADERS, "Referer": f"https://m.sports.naver.com/game/{naver_game_id}/relay"},
        timeout=12,
    )
    response.raise_for_status()
    relay = response.json().get("result", {}).get("textRelayData") or {}

    def lineup(side: str) -> list[dict[str, Any]]:
        batters = (relay.get(f"{side}Lineup") or {}).get("batter") or []
        return [
            {
                "bat_order": int(row.get("batOrder") or 0),
                "name": row.get("name") or "",
                "position": row.get("posName") or "",
                "pcode": str(row.get("pcode") or ""),
            }
            for row in batters
            if row.get("name") and row.get("batOrder")
        ]

    inning_batters: dict[str, list[dict[str, Any]]] = {"away": [], "home": []}
    seen: dict[str, set[str]] = {"away": set(), "home": set()}
    for relay_row in relay.get("textRelays") or []:
        side = "home" if str(relay_row.get("homeOrAway")) == "1" else "away"
        for option in relay_row.get("textOptions") or []:
            batter = option.get("batterRecord") or {}
            player_key = str(batter.get("pcode") or batter.get("name") or "")
            if not player_key or player_key in seen[side] or not batter.get("batOrder"):
                continue
            seen[side].add(player_key)
            inning_batters[side].append({
                "bat_order": int(batter.get("batOrder") or 0),
                "name": batter.get("name") or "",
                "position": batter.get("posName") or "",
                "pcode": str(batter.get("pcode") or ""),
                "_relay_no": int(relay_row.get("no") or 0),
            })

    for rows in inning_batters.values():
        rows.sort(key=lambda row: int(row.get("_relay_no", 0)))
        for row in rows:
            row.pop("_relay_no", None)

    return {
        "game_id": game_id,
        "naver_game_id": naver_game_id,
        "inning": int(relay.get("inn") or inning or 0),
        "away_lineup": lineup("away"),
        "home_lineup": lineup("home"),
        "inning_batters": inning_batters,
        "source": "NAVER Sports public game relay",
    }


def fetch_standings(season: int) -> list[dict[str, Any]]:
    """1982년 이후 정규시즌 팀 순위를 메모리 캐시로 제공한다."""
    key = f"standings:{season}"
    if key in _cache and time.time() - _cache[key][0] < CACHE_TTL:
        return _cache[key][1]
    response = requests.get(
        STANDINGS_URL,
        params={"year": season},
        headers=HEADERS,
        timeout=20,
    )
    response.raise_for_status()
    rows = response.json().get("data", [])
    standings = [
        {
            "rank": int(row.get("rank_position") or row.get("rank_final") or 0),
            "team": row.get("team_name", ""),
            "games": int(row.get("games") or 0),
            "wins": int(row.get("wins") or 0),
            "losses": int(row.get("losses") or 0),
            "draws": int(row.get("draws") or 0),
            "win_rate": float(row.get("win_rate") or 0),
            "games_behind": float(row.get("games_behind") or 0),
            "last_10": row.get("last_10") or "-",
            "streak": row.get("streak") or "-",
        }
        for row in rows
    ]
    standings.sort(key=lambda row: row["rank"])
    if not standings:
        raise RuntimeError(f"{season} 시즌 팀 순위를 찾지 못했습니다.")
    _cache[key] = (time.time(), standings)
    return standings


def fetch_player_matchup(pitcher: str, batter: str) -> dict[str, Any]:
    """현재 투수와 타자의 통산 1:1 상대전적을 조회한다."""
    key = f"matchup:{pitcher}:{batter}"
    if key in _cache and time.time() - _cache[key][0] < 3600:
        return _cache[key][1]  # type: ignore[return-value]
    search = requests.get(
        PLAYER_SEARCH_URL,
        params={"search": pitcher, "limit": 8},
        headers=HEADERS,
        timeout=15,
    )
    search.raise_for_status()
    candidates = search.json().get("data", [])
    player = next(
        (row for row in candidates if row.get("name") == pitcher and row.get("position") == "투수"),
        next((row for row in candidates if row.get("name") == pitcher), None),
    )
    if not player:
        result = {"found": False, "pitcher": pitcher, "batter": batter}
        _cache[key] = (time.time(), result)  # type: ignore[assignment]
        return result
    response = requests.get(
        PLAYER_MATCHUP_URL,
        params={"playerId": player["id"], "period": "career"},
        headers=HEADERS,
        timeout=20,
    )
    response.raise_for_status()
    rows = response.json().get("data", {}).get("all", [])
    row = next((item for item in rows if item.get("opponent_name") == batter), None)
    if not row:
        result = {"found": False, "pitcher": pitcher, "batter": batter}
    else:
        result = {
            "found": True,
            "pitcher": pitcher,
            "batter": batter,
            "pa": int(row.get("pa") or 0),
            "ab": int(row.get("ab") or 0),
            "hits": int(row.get("hits") or 0),
            "hr": int(row.get("hr") or 0),
            "so": int(row.get("so") or 0),
            "bb": int(row.get("bb") or 0),
            "avg": float(row.get("avg") or 0),
            "ops": float(row.get("ops") or 0),
        }
    _cache[key] = (time.time(), result)  # type: ignore[assignment]
    return result
