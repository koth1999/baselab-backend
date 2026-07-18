"""KBO 기록실 HTML 수집기. 영구 저장 없이 10분 메모리 캐시만 사용한다."""
from __future__ import annotations

import re
import math
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.koreabaseball.com/Record/Player"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; BaseLab/1.0; +local-analysis)",
    "Referer": "https://www.koreabaseball.com/Record/Player/HitterBasic/Basic1.aspx",
}
CACHE_TTL = 600
_cache: dict[str, tuple[float, Any]] = {}
ARCHIVE_URL = "https://huggingface.co/datasets/juhonov/KBOresearch/resolve/main/kbo_{kind}_stats_2000_2025.json"
HISTORY_RANKING_URL = "https://www.yagoonara.com/api/rankings"
STANDINGS_URL = "https://www.yagoonara.com/api/standings"
PLAYER_SEARCH_URL = "https://www.yagoonara.com/api/players"
PLAYER_MATCHUP_URL = "https://www.yagoonara.com/api/matchups/player"
PLAYER_DETAIL_URL = "https://www.yagoonara.com/api/players/{player_id}"
TEAM_LIST_URL = "https://www.yagoonara.com/api/teams"
TEAM_DETAIL_URL = "https://www.yagoonara.com/api/teams/{team_id}"

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


def _official_season_soup(url: str, season: int) -> BeautifulSoup:
    """KBO ASP.NET 기록표의 연도 선택 postback을 재현한다."""
    session = requests.Session()
    response = session.get(url, headers={**HEADERS, "Referer": url}, timeout=20)
    response.raise_for_status()
    response.encoding = response.apparent_encoding or "utf-8"
    soup = BeautifulSoup(response.text, "html.parser")
    season_select = next(
        (select for select in soup.select("select") if select.select_one(f'option[value="{season}"]')),
        None,
    )
    if season_select is None:
        raise RuntimeError(f"{season} 시즌 선택 항목을 찾지 못했습니다.")
    data = {
        node.get("name"): node.get("value", "")
        for node in soup.select('input[type="hidden"][name]')
    }
    season_name = season_select.get("name")
    data["__EVENTTARGET"] = season_name
    data[season_name] = str(season)
    for field_name in list(data):
        if field_name.endswith("hfSearchYear"):
            data[field_name] = str(season)
        elif field_name.endswith("hfSearchDate"):
            data[field_name] = f"{season}1231"
    for select in soup.select("select[name]"):
        if select is season_select:
            continue
        selected = select.select_one("option[selected]") or select.select_one("option")
        if selected:
            data[select.get("name")] = selected.get("value", "")
    response = session.post(url, data=data, headers={**HEADERS, "Referer": url}, timeout=25)
    response.raise_for_status()
    response.encoding = response.apparent_encoding or "utf-8"
    return BeautifulSoup(response.text, "html.parser")


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

    player_lookup: dict[str, dict[str, Any]] = {}
    for side_name in ("away", "home"):
        side_lineup = relay.get(f"{side_name}Lineup") or {}
        for role in ("batter", "pitcher"):
            for row in side_lineup.get(role) or []:
                pcode = str(row.get("pcode") or "")
                if pcode:
                    player_lookup[pcode] = {
                        "name": row.get("name") or "",
                        "pcode": pcode,
                        "back_number": str(row.get("backnum") or ""),
                        "throws_bats": row.get("hitType") or "",
                    }

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
    at_bats: dict[str, list[dict[str, Any]]] = {"away": [], "home": []}
    seen: dict[str, set[str]] = {"away": set(), "home": set()}
    for relay_row in relay.get("textRelays") or []:
        side = "home" if str(relay_row.get("homeOrAway")) == "1" else "away"
        options = relay_row.get("textOptions") or []
        batter_option = next((option for option in options if option.get("batterRecord")), None)
        if batter_option:
            batter = batter_option.get("batterRecord") or {}
            pts_by_number = {
                int(point.get("ballcount") or 0): point
                for point in relay_row.get("ptsOptions") or []
                if point.get("ballcount")
            }
            pitches: list[dict[str, Any]] = []
            result = ""
            pitcher_pcode = ""
            for option in options:
                text = str(option.get("text") or "").strip()
                pitch_match = re.match(r"^(\d+)구\s+(.+)$", text)
                if int(option.get("type") or 0) == 1 and pitch_match:
                    pitch_number = int(pitch_match.group(1))
                    point = pts_by_number.get(pitch_number) or {}
                    speed_value = option.get("speed") or option.get("velocity")
                    state = option.get("currentGameState") or {}
                    state_pitcher = state.get("pitcher")
                    pitcher_pcode = str(
                        state_pitcher.get("pcode") if isinstance(state_pitcher, dict) else state_pitcher or pitcher_pcode
                    )
                    plate_x = point.get("crossPlateX")
                    plate_y = point.get("crossPlateY")
                    plate_z = None
                    try:
                        ay, vy0 = float(point["ay"]), float(point["vy0"])
                        distance = float(point["y0"]) - float(plate_y)
                        discriminant = vy0 * vy0 - 2 * ay * distance
                        roots = [(-vy0 + sign * math.sqrt(discriminant)) / ay for sign in (-1, 1)]
                        flight_time = min(root for root in roots if root > 0)
                        plate_z = float(point["z0"]) + float(point["vz0"]) * flight_time + 0.5 * float(point["az"]) * flight_time**2
                    except (KeyError, TypeError, ValueError, ZeroDivisionError):
                        pass
                    top_sz = float(point.get("topSz") or 3.5)
                    bottom_sz = float(point.get("bottomSz") or 1.5)
                    x_percent = max(3, min(97, (float(plate_x) + 1.5) / 3 * 100)) if plate_x is not None else None
                    y_percent = max(3, min(97, (top_sz + 0.8 - plate_z) / (top_sz - bottom_sz + 1.6) * 100)) if plate_z is not None else None
                    pitches.append({
                        "number": pitch_number,
                        "call": pitch_match.group(2).strip(),
                        "pitch_type": option.get("stuff") or option.get("pitchType") or option.get("pitch_type"),
                        "speed": float(speed_value) if speed_value not in (None, "") else None,
                        "x": round(x_percent, 2) if x_percent is not None else None,
                        "y": round(y_percent, 2) if y_percent is not None else None,
                        "plate_x": plate_x,
                        "plate_z": round(plate_z, 3) if plate_z is not None else None,
                        "count": f'{state.get("ball", 0)}-{state.get("strike", 0)}',
                        "kind": "ball" if "볼" in pitch_match.group(2) else "inplay" if "타격" in pitch_match.group(2) else "strike",
                    })
                elif int(option.get("type") or 0) == 13 and text:
                    result = text.split(":", 1)[-1].strip()

            pitches.sort(key=lambda pitch: pitch["number"])
            if pitches or result:
                at_bats[side].append({
                    "relay_no": int(relay_row.get("no") or 0),
                    "bat_order": int(batter.get("batOrder") or 0),
                    "name": batter.get("name") or "",
                    "pcode": str(batter.get("pcode") or ""),
                    "result": result or pitches[-1]["call"],
                    "pitches": pitches,
                    "batter_profile": player_lookup.get(str(batter.get("pcode") or "")),
                    "pitcher": player_lookup.get(pitcher_pcode),
                })

        for option in options:
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

    for rows in at_bats.values():
        rows.sort(key=lambda row: row["relay_no"])
        for row in rows:
            row.pop("relay_no", None)

    return {
        "game_id": game_id,
        "naver_game_id": naver_game_id,
        "inning": int(relay.get("inn") or inning or 0),
        "away_lineup": lineup("away"),
        "home_lineup": lineup("home"),
        "inning_batters": inning_batters,
        "at_bats": at_bats,
        "source": "NAVER Sports public game relay",
    }


def fetch_standings(season: int) -> list[dict[str, Any]]:
    """1982년 이후 정규시즌 팀 순위를 메모리 캐시로 제공한다."""
    key = f"standings:{season}"
    if key in _cache and time.time() - _cache[key][0] < CACHE_TTL:
        return _cache[key][1]
    standings: list[dict[str, Any]] = []
    decade = season // 10 * 10
    response = requests.get(
        "https://www.koreabaseball.com/Record/History/Team/Record.aspx",
        params={"startYear": decade, "halfSc": "T"},
        headers=HEADERS,
        timeout=25,
    )
    response.raise_for_status()
    response.encoding = response.apparent_encoding or "utf-8"
    soup = BeautifulSoup(response.text, "html.parser")
    season_table = next(
        (table for table in soup.select("table") if table.select_one("th") and table.select_one("th").get_text(strip=True) == str(season)),
        None,
    )
    if season_table:
        for rank, row in enumerate(season_table.select("tbody tr"), 1):
            values = [cell.get_text(" ", strip=True) for cell in row.select("th, td")]
            if len(values) < 8 or values[0] == "합계":
                continue
            standings.append({
                "rank": rank, "team": values[0], "games": int(values[1]),
                "wins": int(values[2]), "losses": int(values[3]), "draws": int(values[4]),
                "win_rate": float(values[7]), "games_behind": 0.0,
                "last_10": "-", "streak": "-",
                "team_avg": float(values[5]), "team_era": float(values[6]),
            })
    if standings:
        leader = standings[0]
        for row in standings:
            row["games_behind"] = round(
                ((leader["wins"] - row["wins"]) + (row["losses"] - leader["losses"])) / 2, 1
            )
    standings.sort(key=lambda row: row["rank"])
    if not standings:
        raise RuntimeError(f"{season} 시즌 팀 순위를 찾지 못했습니다.")
    _cache[key] = (time.time(), standings)
    return standings


def fetch_hitter_advanced_rankings(season: int) -> dict[str, dict[str, float | None]]:
    key = f"advanced-hitters:{season}"
    if key in _cache and time.time() - _cache[key][0] < CACHE_TTL:
        return _cache[key][1]

    stat_map = {
        "ops": "OPS",
        "isop": "ISO",
        "babip": "BABIP",
        "woba": "wOBA",
        "wrc_plus": "wRC+",
        "war": "WAR",
    }

    def fetch_stat(item: tuple[str, str]) -> tuple[str, list[dict[str, Any]]]:
        stat, label = item
        response = requests.get(
            HISTORY_RANKING_URL,
            params={
                "type": "hitter",
                "stat": stat,
                "period": season,
                "limit": 1000,
                "qualifyOnly": "false",
            },
            timeout=20,
        )
        try:
            response.raise_for_status()
            return label, response.json().get("data") or []
        except requests.RequestException:
            return label, []

    merged: dict[str, dict[str, float | None]] = {}
    with ThreadPoolExecutor(max_workers=len(stat_map)) as executor:
        for label, rows in executor.map(fetch_stat, stat_map.items()):
            for row in rows:
                player_key = f"{row.get('player_name')}|{row.get('team_name')}"
                try:
                    value = float(row.get("value"))
                except (TypeError, ValueError):
                    value = None
                merged.setdefault(player_key, {})[label] = value
    for values in merged.values():
        values.setdefault("WPA", None)
    _cache[key] = (time.time(), merged)
    return merged


def fetch_team_season_stats(season: int) -> dict[str, dict[str, Any]]:
    key = f"team-season-stats:{season}"
    if key in _cache and time.time() - _cache[key][0] < CACHE_TTL:
        return _cache[key][1]

    if season == datetime.now().year:
        def official_rows(path: str) -> list[list[str]]:
            url = f"https://www.koreabaseball.com/Record/Team/{path}"
            response = requests.get(url, headers=HEADERS, timeout=20)
            response.raise_for_status()
            response.encoding = response.apparent_encoding or "utf-8"
            soup = BeautifulSoup(response.text, "html.parser")
            return [
                [cell.get_text(" ", strip=True) for cell in row.select("th, td")]
                for row in soup.select("table.tData01 tbody tr, table.tData tbody tr")
            ]

        result: dict[str, dict[str, Any]] = {}
        for values in official_rows("Hitter/Basic1.aspx"):
            if len(values) >= 15 and values[0].isdigit():
                result.setdefault(values[1], {}).update({
                    "team_avg": float(values[2]), "team_runs": int(values[6]),
                    "team_hits": int(values[7]), "team_hr": int(values[10]), "team_rbi": int(values[12]),
                })
        for values in official_rows("Hitter/Basic2.aspx"):
            if len(values) >= 11 and values[0].isdigit():
                result.setdefault(values[1], {})["team_ops"] = float(values[10])
        for values in official_rows("Pitcher/Basic1.aspx"):
            if len(values) >= 18 and values[0].isdigit():
                result.setdefault(values[1], {}).update({
                    "team_era": float(values[2]), "team_so": int(values[14]), "team_whip": float(values[17]),
                })
        for values in official_rows("Runner/Basic.aspx"):
            if len(values) >= 6 and values[0].isdigit():
                result.setdefault(values[1], {})["team_sb"] = int(values[4])
        if result:
            _cache[key] = (time.time(), result)
            return result

    if 2000 <= season <= 2025:
        response = requests.get(ARCHIVE_URL.format(kind="hitter"), headers=HEADERS, timeout=30)
        response.raise_for_status()
        result: dict[str, dict[str, Any]] = {}
        totals: dict[str, dict[str, float]] = {}
        for row in response.json().get("data", []):
            if int(row.get("year", 0)) != season:
                continue
            team = str(row.get("팀명") or row.get("teamName") or "")
            bucket = totals.setdefault(team, {key: 0.0 for key in ("AB", "R", "H", "HR", "RBI")})
            for field in bucket:
                bucket[field] += float(row.get(field, 0) or 0)
        for team, total in totals.items():
            result[team] = {
                "team_avg": round(total["H"] / total["AB"], 3) if total["AB"] else 0,
                "team_runs": int(total["R"]), "team_hits": int(total["H"]),
                "team_hr": int(total["HR"]), "team_rbi": int(total["RBI"]),
            }
        if result:
            _cache[key] = (time.time(), result)
            return result

    team_response = requests.get(TEAM_LIST_URL, timeout=20)
    team_response.raise_for_status()
    teams = [
        row for row in team_response.json().get("data") or []
        if row.get("team_type") == "kbo" and row.get("is_current")
    ]

    def fetch_team(team: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        response = requests.get(TEAM_DETAIL_URL.format(team_id=team["id"]), timeout=30)
        response.raise_for_status()
        data = response.json().get("data") or {}

        def season_row(current_key: str, history_key: str) -> dict[str, Any]:
            current = data.get(current_key) or {}
            if int(current.get("year") or 0) == season:
                return current
            return next(
                (row for row in data.get(history_key) or [] if int(row.get("year") or 0) == season),
                {},
            )

        hitter = season_row("teamHitterStats", "teamHitterHistory")
        pitcher = season_row("teamPitcherStats", "teamPitcherHistory")
        running = season_row("teamRunningStats", "teamRunningHistory")
        return team["name"], {
            "team_avg": float(hitter.get("avg") or 0),
            "team_hits": int(hitter.get("hits") or 0),
            "team_hr": int(hitter.get("hr") or 0),
            "team_runs": int(hitter.get("runs") or 0),
            "team_rbi": int(hitter.get("rbi") or 0),
            "team_ops": float(hitter.get("ops") or 0),
            "team_sb": int(running.get("sb") or 0),
            "team_era": float(pitcher.get("era") or 0),
            "team_whip": float(pitcher.get("whip") or 0),
            "team_so": int(pitcher.get("so") or 0),
        }

    with ThreadPoolExecutor(max_workers=10) as executor:
        result = dict(executor.map(fetch_team, teams))
    _cache[key] = (time.time(), result)
    return result


def _fetch_official_player_profile(name: str, team: str, position: str) -> dict[str, Any]:
    """KBO 공식 선수 검색과 상세 기록표를 이용하는 프로필 대체 경로."""
    search = requests.post(
        "https://www.koreabaseball.com/ws/Controls.asmx/GetSearchPlayer",
        data={"name": name},
        headers={**HEADERS, "Referer": "https://www.koreabaseball.com/Player/Search.aspx", "X-Requested-With": "XMLHttpRequest"},
        timeout=20,
    )
    search.raise_for_status()
    payload = search.json()
    candidates = (payload.get("now") or []) + (payload.get("retire") or [])
    want_pitcher = position == "pitcher"

    def score(row: dict[str, Any]) -> tuple[int, int, int]:
        return (
            int(row.get("P_NM") == name),
            int(bool(team) and row.get("T_NM") == team),
            int((row.get("POS_NO") == "투수") == want_pitcher),
        )

    player = max(candidates, key=score, default=None)
    if not player or player.get("P_NM") != name:
        return {"found": False, "name": name, "team": team}

    player_id = str(player.get("P_ID") or "")
    kind = "Pitcher" if player.get("POS_NO") == "투수" else "Hitter"
    base = f"https://www.koreabaseball.com/Record/Player/{kind}Detail"

    def soup_for(leaf: str) -> BeautifulSoup:
        response = requests.get(f"{base}/{leaf}.aspx", params={"playerId": player_id}, headers=HEADERS, timeout=20)
        response.raise_for_status()
        response.encoding = response.apparent_encoding or "utf-8"
        return BeautifulSoup(response.text, "html.parser")

    basic_soup = soup_for("Basic")
    total_soup = soup_for("Total")
    basic = basic_soup.select_one(".player_basic")

    def field(suffix: str) -> str:
        node = basic.select_one(f'[id$="{suffix}"]') if basic else None
        return node.get_text(" ", strip=True) if node else ""

    position_text = field("lblPosition") or str(player.get("POS_NO") or "")
    type_match = re.search(r"\(([^)]+)\)", position_text)
    player_type = type_match.group(1) if type_match else str(player.get("P_TYPE") or "")
    size_match = re.search(r"(\d+)cm/(\d+)kg", field("lblHeightWeight"))
    birth = field("lblBirthday").replace("년 ", "-").replace("월 ", "-").replace("일", "")
    image = basic.select_one("img") if basic else None
    image_url = image.get("src") if image else None
    if image_url and image_url.startswith("//"):
        image_url = "https:" + image_url

    table = total_soup.select_one("table")
    headers = [cell.get_text(" ", strip=True) for cell in table.select("thead th")] if table else []
    seasons: list[dict[str, Any]] = []
    if table:
        for tr in table.select("tbody tr"):
            values = [cell.get_text(" ", strip=True) for cell in tr.select("td")]
            if len(values) != len(headers) or not values[0].isdigit():
                continue
            raw = dict(zip(headers, values))
            if kind == "Hitter":
                obp = float(raw["OBP"])
                slg = float(raw["SLG"])
                pa = int(raw["PA"])
                wrc_plus = max(0.0, 100 * (obp / .335 + slg / .405 - 1))
                war = (wrc_plus - 100) * pa / 600 * .08 + 2 * pa / 600
                seasons.append({
                    "year": int(raw["연도"]), "team_name": raw["팀명"], "games": int(raw["G"]),
                    "pa": int(raw["PA"]), "ab": int(raw["AB"]), "hits": int(raw["H"]),
                    "avg": raw["AVG"], "obp": raw["OBP"], "slg": raw["SLG"],
                    "ops": f'{float(raw["OBP"]) + float(raw["SLG"]):.3f}',
                    "hr": int(raw["HR"]), "rbi": int(raw["RBI"]), "sb": int(raw["SB"]),
                    "bb": int(raw["BB"]), "hbp": int(raw["HBP"]), "tb": int(raw["TB"]),
                    "wrc_plus": round(wrc_plus, 1), "war": round(war, 1),
                })
            else:
                ip_text = raw["IP"]
                ip_parts = ip_text.split()
                innings = float(ip_parts[0]) if ip_parts else 0.0
                if len(ip_parts) > 1:
                    innings += {"1/3": 1 / 3, "2/3": 2 / 3}.get(ip_parts[1], 0)
                whip = (int(raw["H"]) + int(raw["BB"])) / innings if innings else 0
                seasons.append({
                    "year": int(raw["연도"]), "team_name": raw["팀명"], "games": int(raw["G"]),
                    "innings": raw["IP"], "ip": raw["IP"], "era": raw["ERA"],
                    "wins": int(raw["W"]), "losses": int(raw["L"]), "saves": int(raw["SV"]),
                    "holds": int(raw["HLD"]), "so": int(raw["SO"]), "whip": f"{whip:.2f}", "war": None,
                })
    seasons.sort(key=lambda row: row["year"], reverse=True)

    recent_games: list[dict[str, Any]] = []
    tables = basic_soup.select("table")
    if len(tables) >= 3:
        current_year = datetime.now().year
        for tr in tables[2].select("tbody tr")[:5]:
            values = [cell.get_text(" ", strip=True) for cell in tr.select("td")]
            if kind == "Hitter" and len(values) >= 17:
                recent_games.append({
                    "game_date": f"{current_year}-{values[0].replace('.', '-')}", "opponent": values[1],
                    "h_ab": int(values[4]), "h_hits": int(values[6]), "h_hr": int(values[9]),
                    "h_rbi": int(values[10]), "h_so": int(values[15]),
                })

    career: dict[str, Any] = {
        "seasons": len(seasons),
        "first_year": seasons[-1]["year"] if seasons else None,
        "last_year": seasons[0]["year"] if seasons else None,
        "games": sum(int(row.get("games") or 0) for row in seasons),
        "war": None,
    }
    if kind == "Hitter":
        for key_name in ("pa", "ab", "hits", "hr", "rbi", "sb", "bb", "hbp", "tb"):
            career[key_name] = sum(int(row.get(key_name) or 0) for row in seasons)
        career["avg"] = f'{career["hits"] / career["ab"]:.3f}' if career["ab"] else "0.000"
        career["slg"] = f'{career["tb"] / career["ab"]:.3f}' if career["ab"] else "0.000"
        career["obp"] = f'{sum(float(row["obp"]) * int(row["pa"]) for row in seasons) / career["pa"]:.3f}' if career["pa"] else "0.000"
        career["ops"] = f'{float(career["obp"]) + float(career["slg"]):.3f}'
        career["wrc_plus"] = round(
            sum(float(row["wrc_plus"]) * int(row["pa"]) for row in seasons) / career["pa"], 1
        ) if career["pa"] else None
        career["war"] = round(sum(float(row["war"]) for row in seasons), 1)
    else:
        for key_name in ("wins", "losses", "saves", "holds", "so"):
            career[key_name] = sum(int(row.get(key_name) or 0) for row in seasons)
        career.update({"innings": "-", "era": "-", "whip": "-"})

    return {
        "found": True,
        "profile": {
            "kbo_id": player_id, "name": player.get("P_NM"), "name_en": "", "team": player.get("T_NM"),
            "back_number": str(player.get("BACK_NO") or ""), "birth_date": birth,
            "position": player.get("POS_NO"), "primary_pos": position_text.split("(")[0],
            "throws": "좌" if "좌투" in player_type else "우", "bats": "좌" if "좌타" in player_type else "양" if "양타" in player_type else "우",
            "height": int(size_match.group(1)) if size_match else 0, "weight": int(size_match.group(2)) if size_match else 0,
            "career_history": field("lblCareer"), "draft_info": field("lblDraft"),
            "debut_year": seasons[-1]["year"] if seasons else None, "image_url": image_url,
        },
        "seasons": seasons, "career": career, "recent_games": recent_games,
        "source": "KBO official player records",
    }


def fetch_player_profile(name: str, team: str = "", position: str = "hitter") -> dict[str, Any]:
    """선수 기본정보, KBO 정규시즌 연도별 기록, 고급지표와 최근 5경기를 반환한다."""
    try:
        search_response = requests.get(PLAYER_SEARCH_URL, params={"search": name, "limit": 20}, timeout=20)
        search_response.raise_for_status()
    except requests.RequestException:
        return _fetch_official_player_profile(name, team, position)
    candidates = search_response.json().get("data") or []

    def candidate_score(row: dict[str, Any]) -> tuple[int, int, int]:
        exact_name = int(row.get("name") == name)
        exact_team = int(bool(team) and row.get("team_name") == team)
        is_pitcher = row.get("position") == "투수"
        exact_position = int(is_pitcher == (position == "pitcher"))
        return exact_name, exact_team, exact_position

    candidate = max(candidates, key=candidate_score, default=None)
    if not candidate or candidate.get("name") != name:
        return {"found": False, "name": name, "team": team}

    try:
        detail_response = requests.get(PLAYER_DETAIL_URL.format(player_id=candidate["id"]), timeout=30)
        detail_response.raise_for_status()
    except requests.RequestException:
        return _fetch_official_player_profile(name, team, position)
    detail = detail_response.json().get("data") or {}
    is_pitcher = detail.get("position") == "투수"
    stat_key = "pitcher_stats" if is_pitcher else "hitter_stats"
    career_key = "pitcher_career_advanced" if is_pitcher else "hitter_career_advanced"
    seasons = [
        row for row in detail.get(stat_key) or []
        if row.get("league_type") == "kbo" and row.get("game_type") == "regular"
    ]
    seasons.sort(key=lambda row: int(row.get("year") or 0), reverse=True)
    recent_games = [
        row for row in detail.get("recent_games") or []
        if row.get("game_type") == "regular"
    ]
    recent_games.sort(key=lambda row: row.get("game_date") or "", reverse=True)
    kbo_id = str(detail.get("kbo_id") or "")

    return {
        "found": True,
        "profile": {
            "id": detail.get("id"),
            "kbo_id": kbo_id,
            "name": detail.get("name"),
            "name_en": detail.get("name_en"),
            "team": detail.get("team_name"),
            "back_number": detail.get("back_number"),
            "birth_date": detail.get("birth_date"),
            "position": detail.get("position"),
            "primary_pos": detail.get("primary_pos"),
            "throws": detail.get("throws"),
            "bats": detail.get("bats"),
            "height": detail.get("height"),
            "weight": detail.get("weight"),
            "career_history": detail.get("career_history"),
            "draft_info": detail.get("draft_info"),
            "debut_year": (seasons[-1].get("year") if seasons else detail.get("debut_year")),
            "image_url": detail.get("custom_image_url") or (
                f"https://www.yagoonara.com/players/{kbo_id}.jpg" if kbo_id else None
            ),
        },
        "seasons": seasons,
        "career": detail.get(career_key) or {},
        "recent_games": recent_games[:5],
        "source": "yagoonara player database",
    }


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
