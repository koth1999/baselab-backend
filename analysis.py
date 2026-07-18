"""수집한 기본 기록을 사용자 친화적인 점수와 스카우팅 문장으로 변환한다."""
from __future__ import annotations

from typing import Any


def _percentile(values: list[float], value: float, higher_is_better: bool = True) -> float:
    if not values:
        return 50
    rank = sum(v <= value for v in values) / len(values) * 100
    return rank if higher_is_better else 100 - rank


def _grade(score: int) -> str:
    return "S" if score >= 95 else "A+" if score >= 90 else "A" if score >= 82 else "B+" if score >= 74 else "B" if score >= 65 else "C"


def _hitter_metrics(player: dict[str, Any]) -> dict[str, float]:
    """기본 타격 기록으로 계산 가능한 세이버 지표와 기여도 추정치를 만든다."""
    ab, hits, bb, hbp, so, pa = (
        float(player.get(key, 0) or 0)
        for key in ("AB", "H", "BB", "HBP", "SO", "PA")
    )
    doubles, triples, hr, sb, cs = (
        float(player.get(key, 0) or 0)
        for key in ("2B", "3B", "HR", "SB", "CS")
    )
    singles = max(0.0, hits - doubles - triples - hr)
    sf = max(0.0, pa - ab - bb - hbp)
    avg = hits / ab if ab else 0.0
    slg = (singles + 2 * doubles + 3 * triples + 4 * hr) / ab if ab else 0.0
    babip_denominator = ab - so - hr + sf
    woba_denominator = ab + bb + hbp + sf
    return {
        "ISO": max(0.0, slg - avg),
        "BABIP": (hits - hr) / babip_denominator if babip_denominator > 0 else 0.0,
        "wOBA": (
            .69 * bb + .72 * hbp + .89 * singles + 1.27 * doubles + 1.62 * triples + 2.10 * hr
        ) / woba_denominator if woba_denominator > 0 else 0.0,
    }


def analyze(players: list[dict[str, Any]], position: str) -> list[dict[str, Any]]:
    calculated_hitters = [_hitter_metrics(player) for player in players] if position == "hitter" else []
    league_woba = (
        sum(metric["wOBA"] * float(player.get("PA", 0) or 0) for metric, player in zip(calculated_hitters, players))
        / sum(float(player.get("PA", 0) or 0) for player in players)
        if players and sum(float(player.get("PA", 0) or 0) for player in players) > 0
        else .320
    )
    analyzed = []
    for player_index, player in enumerate(players):
        if position == "hitter":
            ab, hits, bb, hbp, sf = (float(player.get(k, 0) or 0) for k in ("AB", "H", "BB", "HBP", "SF"))
            doubles, triples, hr = (float(player.get(k, 0) or 0) for k in ("2B", "3B", "HR"))
            avg = float(player.get("AVG", 0) or 0)
            obp = (hits + bb + hbp) / (ab + bb + hbp + sf) if ab + bb + hbp + sf else avg
            slg = (hits + doubles + 2 * triples + 3 * hr) / ab if ab else 0
            calculated_ops = obp + slg
            ops = float(player.get("OPS", 0) or 0) or calculated_ops
            score = round(.45 * _percentile([float(p.get("AVG", 0) or 0) for p in players], avg) + .35 * _percentile([float(p.get("HR", 0) or 0) for p in players], hr) + .2 * _percentile([float(p.get("RBI", 0) or 0) for p in players], float(player.get("RBI", 0) or 0)))
            strengths = (["정교한 컨택"] if avg >= .300 else []) + (["엘리트 장타 생산"] if hr >= 15 else []) + (["높은 출루 생산성"] if obp >= .380 else [])
            weaknesses = (["장타 생산 보완"] if slg < .400 else []) + (["출루율 개선"] if obp < .330 else [])
            stats = {key: player.get(key, 0) or 0 for key in ("G", "PA", "AB", "H", "2B", "3B", "HR", "RBI", "SB", "CS", "BB", "HBP", "SO", "GDP", "E")}
            stats.update({"AVG": avg, "OBP": round(obp, 3), "SLG": round(slg, 3), "OPS": round(ops, 3)})
            calculated = calculated_hitters[player_index]
            pa = float(player.get("PA", 0) or 0)
            sb = float(player.get("SB", 0) or 0)
            cs = float(player.get("CS", 0) or 0)
            batting_runs = (calculated["wOBA"] - league_woba) / 1.25 * pa
            base_running_runs = .2 * (sb - .4 * cs)
            replacement_runs = 20 * pa / 600
            estimates = {
                "ISO": round(calculated["ISO"], 3),
                "BABIP": round(calculated["BABIP"], 3),
                "wOBA": round(calculated["wOBA"], 3),
                "wRC+": round(100 * calculated["wOBA"] / league_woba, 1) if league_woba else 100.0,
                "WPA": round(batting_runs / 10, 2),
                "WAR": round((batting_runs + base_running_runs + replacement_runs) / 10, 1),
            }
            stats.update({
                key: player.get(key) if player.get(key) is not None else estimates[key]
                for key in ("ISO", "BABIP", "wOBA", "wRC+", "WPA", "WAR")
            })
            summary = f"타율 {avg:.3f}, OPS {ops:.3f}을 기록 중인 {player['team']}의 공격 자원입니다."
        else:
            era, whip, wins, so, ip = (float(player.get(k, 0) or 0) for k in ("ERA", "WHIP", "W", "SO", "IP"))
            score = round(.4 * _percentile([float(p.get("ERA", 99) or 99) for p in players], era, False) + .35 * _percentile([float(p.get("WHIP", 9) or 9) for p in players], whip, False) + .25 * _percentile([float(p.get("SO", 0) or 0) for p in players], so))
            strengths = (["뛰어난 실점 억제"] if era <= 3.5 else []) + (["안정적인 출루 억제"] if whip <= 1.25 else []) + (["탈삼진 능력"] if so >= 70 else [])
            weaknesses = (["볼넷·출루 허용 관리"] if whip >= 1.5 else []) + (["실점 억제 개선"] if era >= 5 else [])
            stats = {key: player.get(key, 0) or 0 for key in ("G", "W", "L", "SV", "HLD", "WPCT", "IP", "H", "HR", "BB", "HBP", "SO", "R", "ER")}
            stats.update({"ERA": era, "WHIP": whip})
            summary = f"평균자책점 {era:.2f}, WHIP {whip:.2f}를 기록 중인 {player['team']} 투수입니다."
        analyzed.append({"rank": player.get("rank", 0), "name": player["name"], "team": player["team"], "position": player["position"], "stats": stats, "score": max(20, min(99, score)), "grade": _grade(score), "strengths": strengths or ["평균 이상의 종합 기여"], "weaknesses": weaknesses or ["뚜렷한 약점 없음"], "summary": summary})
    return sorted(analyzed, key=lambda item: item["score"], reverse=True)
