"""
NBA Question Answering System using Hybrid Approach (Strategy 2)

This module implements a RAG-like system that:
1. Analyzes questions to extract entities (players, teams, dates, stats)
2. Queries NBA API for structured data
3. Converts API responses to natural language context
4. Uses Hugging Face QA model to answer questions from context
5. Formats answers with citations

Author: Generated for NBA Chat Assistant
"""

import re
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass

from transformers import pipeline
from nba_api.stats.endpoints import (
    playercareerstats,
    playergamelog,
    leaguegamefinder,
    boxscoretraditionalv2,
    playbyplay,
    commonplayerinfo,
    teamgamelog,
    leaguedashplayerstats,
    leagueleaders,
)
from nba_api.stats.static import teams, players
from nba_api.stats.library.parameters import Season, SeasonType


@dataclass
class QuestionAnalysis:
    """Structured analysis of a user question"""

    question: str
    question_type: (
        str  # 'player_stats', 'game_result', 'comparison', 'play_by_play', 'general'
    )
    entities: Dict[str, Any]  # Extracted entities (players, teams, dates, stats)
    confidence: float


@dataclass
class QAAnswer:
    """Structured answer from QA system"""

    answer: str
    confidence: float
    context_used: str
    sources: List[Dict[str, str]]  # API endpoints, game IDs, dates
    raw_data: Optional[Dict] = None


class QuestionAnalyzer:
    """Analyzes questions to extract entities and determine question type"""

    # Common stat keywords
    STAT_KEYWORDS = {
        "points": ["points", "scored", "scoring", "pts"],
        "rebounds": ["rebounds", "rebound", "rebs", "boards"],
        "assists": ["assists", "assist", "ast"],
        "steals": ["steals", "steal", "stl"],
        "blocks": ["blocks", "block", "blk"],
        "turnovers": ["turnovers", "turnover", "tov"],
        "three_pointers": [
            "three pointers",
            "3 pointers",
            "3-pointers",
            "3pt made",
            "three point field goals",
            "3pt field goals",
            "threes made",
        ],
        "field_goal_percentage": [
            "field goal percentage",
            "fg%",
            "shooting percentage",
            "fg percent",
        ],
        "three_point_percentage": [
            "three point percentage",
            "3pt%",
            "3-point percentage",
        ],
        "free_throw_percentage": ["free throw percentage", "ft%", "ft percent"],
        "minutes": ["minutes", "playing time", "min"],
        "average": ["average", "avg", "per game", "ppg", "rpg", "apg"],
    }

    # Temporal keywords
    TEMPORAL_KEYWORDS = {
        "last": ["last", "most recent", "previous", "latest"],
        "this_season": ["this season", "current season", "2024-25", "2025"],
        "career": ["career", "all-time", "all time", "lifetime"],
        "season": ["season", "year"],
    }

    def __init__(self):
        self.nba_teams = teams.get_teams()
        self.nba_players = players.get_players()
        self.team_names = {team["full_name"].lower(): team for team in self.nba_teams}
        self.team_abbrevs = {
            team["abbreviation"].lower(): team for team in self.nba_teams
        }
        self.player_names = {
            player["full_name"].lower(): player for player in self.nba_players
        }

    def analyze(self, question: str) -> QuestionAnalysis:
        """Main analysis method"""
        question_lower = question.lower()

        # Extract entities
        entities = {
            "players": self._extract_players(question_lower),
            "teams": self._extract_teams(question_lower),
            "stats": self._extract_stats(question_lower),
            "temporal": self._extract_temporal(question_lower),
            "comparison": "vs" in question_lower
            or "versus" in question_lower
            or "compare" in question_lower,
            "game_id": self._extract_game_id(question_lower),
            "top_n": self._extract_top_n(question_lower),
            "league_leader": self._is_league_leader_question(question_lower),
        }

        # Determine question type
        question_type = self._classify_question(question_lower, entities)

        # Calculate confidence based on entity extraction
        confidence = self._calculate_confidence(entities)

        return QuestionAnalysis(
            question=question,
            question_type=question_type,
            entities=entities,
            confidence=confidence,
        )

    def _extract_players(self, question: str) -> List[Dict]:
        """
        Extract player names from question (question is already lowercase).
        Prioritizes full name matches over last name matches to avoid ambiguity.
        """
        # First pass: Check for full name matches (prioritize exact matches)
        full_name_matches = []
        for player_name, player_data in self.player_names.items():
            # player_name is already lowercase from initialization
            first_name = player_data["first_name"].lower()
            last_name = player_data["last_name"].lower()

            # Check if full name (first + last) appears together in question
            if player_name in question:
                full_name_matches.append(player_data)
            # Also check if first name and last name appear together (handles variations)
            elif first_name in question and last_name in question:
                # Verify they appear close together (within reasonable distance)
                first_idx = question.find(first_name)
                last_idx = question.find(last_name)
                if (
                    first_idx != -1
                    and last_idx != -1
                    and abs(first_idx - last_idx) < 30
                ):
                    full_name_matches.append(player_data)

        # If we found full name matches, return only those (prioritize specificity)
        if full_name_matches:
            return full_name_matches

        # Second pass: Fallback to last name only if no full name match found
        # This handles cases like "How many points did Curry score?" (ambiguous)
        last_name_matches = []
        seen_last_names = set()
        for player_name, player_data in self.player_names.items():
            last_name = player_data["last_name"].lower()
            # Only check last name if it's a distinct word (not part of another word)
            # Use word boundaries: check if last name appears as a standalone word
            if last_name in question and last_name not in seen_last_names:
                # Verify it's not part of a longer word using word boundaries
                last_name_pattern = r"\b" + re.escape(last_name) + r"\b"
                if re.search(last_name_pattern, question):
                    last_name_matches.append(player_data)
                    seen_last_names.add(last_name)

        return last_name_matches

    def _extract_teams(self, question: str) -> List[Dict]:
        """Extract team names from question"""
        found_teams = []

        # Common team name aliases (short names -> full names)
        team_aliases = {
            "lakers": "los angeles lakers",
            "celtics": "boston celtics",
            "warriors": "golden state warriors",
            "heat": "miami heat",
            "spurs": "san antonio spurs",
            "knicks": "new york knicks",
            "bulls": "chicago bulls",
            "mavs": "dallas mavericks",
            "mavericks": "dallas mavericks",
            "nets": "brooklyn nets",
            "clippers": "los angeles clippers",
            "suns": "phoenix suns",
            "nuggets": "denver nuggets",
            "bucks": "milwaukee bucks",
            "sixers": "philadelphia 76ers",
            "76ers": "philadelphia 76ers",
            "raptors": "toronto raptors",
            "wizards": "washington wizards",
            "hawks": "atlanta hawks",
            "hornets": "charlotte hornets",
            "cavs": "cleveland cavaliers",
            "cavaliers": "cleveland cavaliers",
            "pistons": "detroit pistons",
            "pacers": "indiana pacers",
            "grizzlies": "memphis grizzlies",
            "timberwolves": "minnesota timberwolves",
            "pelicans": "new orleans pelicans",
            "thunder": "oklahoma city thunder",
            "magic": "orlando magic",
            "blazers": "portland trail blazers",
            "trail blazers": "portland trail blazers",
            "kings": "sacramento kings",
            "jazz": "utah jazz",
        }

        # First, check for common aliases (prioritize these)
        question_lower = question.lower()
        for alias, full_name in team_aliases.items():
            # Use word boundaries to avoid partial matches
            alias_pattern = r"\b" + re.escape(alias) + r"\b"
            if re.search(alias_pattern, question_lower):
                # Find the team by full name
                if full_name in self.team_names:
                    found_teams.append(self.team_names[full_name])
                    return found_teams  # Return immediately for alias matches

        # Check full names (exact match preferred)
        for team_name, team_data in self.team_names.items():
            # Use word boundaries for better matching
            team_pattern = r"\b" + re.escape(team_name) + r"\b"
            if re.search(team_pattern, question_lower):
                found_teams.append(team_data)

        # Check abbreviations
        for abbrev, team_data in self.team_abbrevs.items():
            abbrev_pattern = r"\b" + re.escape(abbrev) + r"\b"
            if (
                re.search(abbrev_pattern, question_lower)
                and team_data not in found_teams
            ):
                found_teams.append(team_data)

        return found_teams

    def _extract_stats(self, question: str) -> List[str]:
        """Extract stat keywords from question"""
        found_stats = []
        for stat, keywords in self.STAT_KEYWORDS.items():
            if any(keyword in question for keyword in keywords):
                found_stats.append(stat)
        return found_stats

    def _extract_temporal(self, question: str) -> Dict[str, Any]:
        """Extract temporal information"""
        temporal = {}
        for key, keywords in self.TEMPORAL_KEYWORDS.items():
            if any(keyword in question for keyword in keywords):
                temporal[key] = True

        # Extract specific dates (simple pattern)
        date_pattern = r"\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{4}"
        dates = re.findall(date_pattern, question)
        if dates:
            temporal["specific_date"] = dates[0]

        return temporal

    def _extract_game_id(self, question: str) -> Optional[str]:
        """Extract game ID if present (format: 0022400928)"""
        game_id_pattern = r"00\d{8}"
        match = re.search(game_id_pattern, question)
        return match.group(0) if match else None

    def _extract_top_n(self, question: str) -> Optional[int]:
        """Extract 'top N' number from question (e.g., 'top 5', 'top 10')"""
        # Patterns: "top 5", "top 10", "top 3", etc.
        patterns = [
            r"top\s+(\d+)",
            r"(\d+)\s+best",
            r"(\d+)\s+most",
            r"first\s+(\d+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, question, re.IGNORECASE)
            if match:
                return int(match.group(1))
        return None

    def _is_league_leader_question(self, question: str) -> bool:
        """Check if question is about league leaders/records"""
        leader_keywords = [
            "top",
            "most",
            "best",
            "leader",
            "leaders",
            "record",
            "records",
            "history",
            "all time",
            "all-time",
            "league",
        ]
        return any(keyword in question for keyword in leader_keywords)

    def _classify_question(self, question: str, entities: Dict) -> str:
        """Classify question type"""
        if entities.get("comparison"):
            return "comparison"
        if entities.get("league_leader") and entities.get("stats"):
            return "league_leaders"
        if "play by play" in question or "shot" in question or "play" in question:
            return "play_by_play"
        if entities.get("players") and entities.get("stats"):
            return "player_stats"
        if entities.get("teams") and ("game" in question or "score" in question):
            return "game_result"
        if entities.get("game_id"):
            return "game_result"
        return "general"

    def _calculate_confidence(self, entities: Dict) -> float:
        """Calculate confidence in entity extraction"""
        score = 0.0
        if entities.get("players"):
            score += 0.3
        if entities.get("teams"):
            score += 0.2
        if entities.get("stats"):
            score += 0.2
        if entities.get("temporal"):
            score += 0.2
        if entities.get("game_id"):
            score += 0.1
        return min(score, 1.0)


class NBADataRetriever:
    """Retrieves data from NBA API based on question analysis"""

    def __init__(self):
        self.cache = {}  # Simple cache for API responses
        self.nba_teams = teams.get_teams()  # For team name lookups

    def _get_stat_abbreviation(self, stat_key: str) -> str:
        """Map stat keyword to NBA API abbreviation"""
        stat_map = {
            "points": "PTS",
            "rebounds": "REB",
            "assists": "AST",
            "steals": "STL",
            "blocks": "BLK",
            "turnovers": "TOV",
            "three_pointers": "FG3M",  # 3-point field goals made
            "field_goal_percentage": "FG_PCT",
            "three_point_percentage": "FG3_PCT",
            "free_throw_percentage": "FT_PCT",
        }
        return stat_map.get(stat_key, "PTS")  # Default to points

    def retrieve(self, analysis: QuestionAnalysis) -> Dict[str, Any]:
        """Retrieve relevant NBA data based on question analysis"""
        question_type = analysis.question_type
        entities = analysis.entities

        if question_type == "player_stats":
            return self._get_player_stats(entities)
        elif question_type == "game_result":
            return self._get_game_data(entities)
        elif question_type == "comparison":
            return self._get_comparison_data(entities)
        elif question_type == "play_by_play":
            return self._get_play_by_play(entities)
        elif question_type == "league_leaders":
            return self._get_league_leaders(entities)
        else:
            return self._get_general_data(entities)

    def _get_player_stats(self, entities: Dict) -> Dict[str, Any]:
        """Get player statistics"""
        data = {"type": "player_stats", "players": []}

        for player in entities.get("players", []):
            player_id = player["id"]
            player_name = player["full_name"]

            # Get career stats
            try:
                career_stats = playercareerstats.PlayerCareerStats(player_id=player_id)
                career_df = career_stats.get_data_frames()[0]

                # Get recent game log
                game_log = playergamelog.PlayerGameLog(
                    player_id=player_id, season=Season.default
                )
                game_log_df = game_log.get_data_frames()[0]

                data["players"].append(
                    {
                        "player_id": player_id,
                        "player_name": player_name,
                        "career_stats": career_df.to_dict("records")
                        if not career_df.empty
                        else [],
                        "game_log": game_log_df.to_dict("records")
                        if not game_log_df.empty
                        else [],
                        "recent_games": game_log_df.head(5).to_dict("records")
                        if not game_log_df.empty
                        else [],
                    }
                )
            except Exception as e:
                print(f"Error fetching stats for {player_name}: {e}")
                continue

        return data

    def _get_game_data(self, entities: Dict) -> Dict[str, Any]:
        """Get game data"""
        data = {"type": "game_data", "games": []}

        game_id = entities.get("game_id")
        if game_id:
            # Get specific game
            try:
                boxscore = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id)
                boxscore_df = boxscore.get_data_frames()[0]
                data["games"].append(
                    {
                        "game_id": game_id,
                        "boxscore": boxscore_df.to_dict("records")
                        if not boxscore_df.empty
                        else [],
                    }
                )
            except Exception as e:
                print(f"Error fetching game {game_id}: {e}")
        else:
            # Find games by teams or players
            team_ids = [team["id"] for team in entities.get("teams", [])]
            player_ids = [player["id"] for player in entities.get("players", [])]

            # Check if asking for "last" game (temporal keyword)
            is_last_game = entities.get("temporal", {}).get("last", False)

            try:
                # For team "last game" questions, use LeagueGameFinder without season filter
                # to get all games, then sort by date
                if team_ids and is_last_game:
                    import pandas as pd

                    # Get the team name from entities (the team asked about)
                    # Use the first team found (should be the most relevant)
                    team_entity = entities.get("teams", [])[0]
                    team_name = team_entity.get("full_name", "")
                    team_id = team_entity.get("id")

                    # Verify we have the correct team
                    if not team_name or not team_id:
                        print(f"Warning: Invalid team entity: {team_entity}")
                        return data

                    gamefinder = leaguegamefinder.LeagueGameFinder(
                        team_id_nullable=team_id
                    )
                    games_df = gamefinder.get_data_frames()[0]

                    if not games_df.empty:
                        # Sort by game date (descending) to get most recent
                        games_df["GAME_DATE"] = pd.to_datetime(games_df["GAME_DATE"])
                        games_df = games_df.sort_values("GAME_DATE", ascending=False)

                        # Get most recent game
                        last_game = games_df.iloc[0]
                        game_id = last_game["GAME_ID"]
                        matchup = last_game.get("MATCHUP", "")

                        # Calculate scores
                        team_pts = last_game.get("PTS", 0)
                        plus_minus = last_game.get("PLUS_MINUS", 0)
                        # Opponent score: team_pts - PLUS_MINUS
                        # PLUS_MINUS is the point differential (team_score - opponent_score)
                        opp_pts = team_pts - plus_minus

                        # Extract opponent name from matchup
                        # Matchup format: "LAL vs. WAS" (home) or "LAL @ WAS" (away)
                        matchup_parts = matchup.split()
                        if len(matchup_parts) >= 3:
                            # Get opponent (last part after "vs." or "@")
                            opp_abbrev = matchup_parts[-1]
                            # Try to find full team name from abbreviation
                            opp_team = None
                            for team in self.nba_teams:
                                if team["abbreviation"] == opp_abbrev:
                                    opp_team = team.get("full_name", opp_abbrev)
                                    break
                            opponent_name = opp_team if opp_team else opp_abbrev
                        else:
                            opponent_name = "Opponent"

                        # Get boxscore for additional details
                        try:
                            boxscore = boxscoretraditionalv2.BoxScoreTraditionalV2(
                                game_id=game_id
                            )
                            boxscore_df = boxscore.get_data_frames()[0]
                        except Exception:
                            boxscore_df = pd.DataFrame()

                        data["games"].append(
                            {
                                "game_id": game_id,
                                "game_date": last_game["GAME_DATE"].strftime("%Y-%m-%d")
                                if hasattr(last_game["GAME_DATE"], "strftime")
                                else str(last_game.get("GAME_DATE", "")),
                                "matchup": matchup,
                                "team_name": team_name,  # Store the team asked about
                                "opponent_name": opponent_name,
                                "team_score": int(team_pts),
                                "opponent_score": int(opp_pts),
                                "result": last_game.get("WL", ""),
                                "boxscore": boxscore_df.to_dict("records")
                                if not boxscore_df.empty
                                else [],
                            }
                        )
                else:
                    # Default behavior: find games by teams or players
                    gamefinder = leaguegamefinder.LeagueGameFinder(
                        team_id_nullable=team_ids[0] if team_ids else None,
                        player_id_nullable=player_ids[0] if player_ids else None,
                        season_nullable=Season.default,
                    )
                    games_df = gamefinder.get_data_frames()[0]

                    # Get most recent game
                    if not games_df.empty:
                        recent_game = games_df.iloc[0]
                        game_id = recent_game["GAME_ID"]
                        boxscore = boxscoretraditionalv2.BoxScoreTraditionalV2(
                            game_id=game_id
                        )
                        boxscore_df = boxscore.get_data_frames()[0]

                        data["games"].append(
                            {
                                "game_id": game_id,
                                "game_date": recent_game.get("GAME_DATE", ""),
                                "matchup": recent_game.get("MATCHUP", ""),
                                "boxscore": boxscore_df.to_dict("records")
                                if not boxscore_df.empty
                                else [],
                            }
                        )
            except Exception as e:
                print(f"Error finding games: {e}")

        return data

    def _get_comparison_data(self, entities: Dict) -> Dict[str, Any]:
        """Get data for comparison questions"""
        data = {"type": "comparison", "players": []}

        # Get stats for each player
        for player in entities.get("players", []):
            player_id = player["id"]
            try:
                career_stats = playercareerstats.PlayerCareerStats(player_id=player_id)
                career_df = career_stats.get_data_frames()[0]

                data["players"].append(
                    {
                        "player_id": player_id,
                        "player_name": player["full_name"],
                        "career_stats": career_df.to_dict("records")
                        if not career_df.empty
                        else [],
                    }
                )
            except Exception as e:
                print(f"Error fetching comparison data for {player['full_name']}: {e}")

        return data

    def _get_play_by_play(self, entities: Dict) -> Dict[str, Any]:
        """Get play-by-play data"""
        data = {"type": "play_by_play", "plays": []}

        game_id = entities.get("game_id")
        if not game_id:
            # Try to find recent game
            player_ids = [player["id"] for player in entities.get("players", [])]
            if player_ids:
                try:
                    gamefinder = leaguegamefinder.LeagueGameFinder(
                        player_id_nullable=player_ids[0], season_nullable=Season.default
                    )
                    games_df = gamefinder.get_data_frames()[0]
                    if not games_df.empty:
                        game_id = games_df.iloc[0]["GAME_ID"]
                except Exception as e:
                    print(f"Error finding game for play-by-play: {e}")

        if game_id:
            try:
                pbp = playbyplay.PlayByPlay(game_id=game_id)
                pbp_df = pbp.get_data_frames()[0]
                data["plays"] = pbp_df.to_dict("records") if not pbp_df.empty else []
                data["game_id"] = game_id
            except Exception as e:
                print(f"Error fetching play-by-play: {e}")

        return data

    def _get_league_leaders(self, entities: Dict) -> Dict[str, Any]:
        """Get league leaders data"""
        data = {"type": "league_leaders", "leaders": []}

        # Extract stat type and top N
        stats = entities.get("stats", [])
        top_n = entities.get("top_n", 5)  # Default to top 5

        if not stats:
            return data

        # Get the first stat mentioned (most relevant)
        stat_key = stats[0]
        stat_abbrev = self._get_stat_abbreviation(stat_key)

        try:
            # Request all-time leaders
            leaders = leagueleaders.LeagueLeaders(
                league_id="00",
                season="All Time",
                season_type_all_star="Regular Season",
                stat_category_abbreviation=stat_abbrev,
                per_mode48="Totals",
            )
            df = leaders.get_data_frames()[0]

            if not df.empty:
                # Get top N players
                top_players = df.head(top_n)

                for idx, (_, row) in enumerate(top_players.iterrows(), start=1):
                    # Try to get rank from stat-specific column (e.g., AST_RANK) or generic RANK
                    rank = row.get(f"{stat_abbrev}_RANK") or row.get("RANK") or idx
                    player_name = row.get("PLAYER_NAME", "Unknown")
                    stat_value = row.get(stat_abbrev, 0)

                    data["leaders"].append(
                        {
                            "rank": int(rank),
                            "player_name": player_name,
                            "stat_value": stat_value,
                            "stat_type": stat_key,
                            "stat_abbrev": stat_abbrev,
                        }
                    )
        except Exception as e:
            print(f"Error fetching league leaders: {e}")

        return data

    def _get_general_data(self, entities: Dict) -> Dict[str, Any]:
        """Get general data based on entities"""
        # Fallback to player stats if players are mentioned
        if entities.get("players"):
            return self._get_player_stats(entities)
        return {"type": "general", "data": {}}


class ContextGenerator:
    """Converts structured NBA API data to natural language context"""

    def generate(
        self, data: Dict[str, Any], question: str
    ) -> Tuple[str, List[Dict[str, str]]]:
        """Generate natural language context from API data"""
        data_type = data.get("type", "general")
        sources = []

        if data_type == "player_stats":
            context, sources = self._generate_player_stats_context(data, question)
        elif data_type == "game_data":
            context, sources = self._generate_game_context(data, question)
        elif data_type == "comparison":
            context, sources = self._generate_comparison_context(data, question)
        elif data_type == "play_by_play":
            context, sources = self._generate_play_by_play_context(data, question)
        elif data_type == "league_leaders":
            context, sources = self._generate_league_leaders_context(data, question)
        else:
            context = "No specific NBA data found to answer this question."
            sources = []

        return context, sources

    def _generate_player_stats_context(
        self, data: Dict, question: str
    ) -> Tuple[str, List[Dict]]:
        """Generate context for player statistics"""
        context_parts = []
        sources = []

        for player_data in data.get("players", []):
            player_name = player_data["player_name"]
            player_id = player_data["player_id"]

            # Career stats summary
            career_stats = player_data.get("career_stats", [])
            if career_stats:
                # Calculate career totals and averages (matching training data format)
                total_seasons = len(career_stats)

                # Sum totals across all seasons
                total_pts = sum(season.get("PTS", 0) for season in career_stats)
                total_reb = sum(season.get("REB", 0) for season in career_stats)
                total_ast = sum(season.get("AST", 0) for season in career_stats)
                total_tov = sum(season.get("TOV", 0) for season in career_stats)
                total_blk = sum(season.get("BLK", 0) for season in career_stats)
                total_gp = sum(season.get("GP", 0) for season in career_stats)

                # Calculate career averages
                if total_gp > 0:
                    career_ppg = total_pts / total_gp
                    career_rpg = total_reb / total_gp
                    career_apg = total_ast / total_gp
                    career_tov = total_tov / total_gp
                    career_bpg = total_blk / total_gp
                else:
                    career_ppg = career_rpg = career_apg = career_tov = career_bpg = 0.0

                # Get most recent season for additional context
                latest_season = career_stats[0]  # Most recent season
                season_id = latest_season.get("SEASON_ID", "N/A")
                season_gp = latest_season.get("GP", 0)

                # Calculate season averages (not totals)
                if season_gp > 0:
                    season_pts = latest_season.get("PTS", 0)
                    season_reb = latest_season.get("REB", 0)
                    season_ast = latest_season.get("AST", 0)
                    season_tov = latest_season.get("TOV", 0)
                    season_blk = latest_season.get("BLK", 0)
                    season_ppg = season_pts / season_gp
                    season_rpg = season_reb / season_gp
                    season_apg = season_ast / season_gp
                    season_tov_avg = season_tov / season_gp
                    season_bpg = season_blk / season_gp
                else:
                    season_ppg = season_rpg = season_apg = season_tov_avg = (
                        season_bpg
                    ) = 0.0

                # Generate context matching training data format
                context_parts.append(
                    f"{player_name} has played {total_seasons} seasons in the NBA. "
                    f"Over his career, he has averaged {career_ppg:.1f} points per game, "
                    f"{career_rpg:.1f} rebounds per game, {career_apg:.1f} assists per game, "
                    f"{career_tov:.1f} turnovers per game, and {career_bpg:.1f} blocks per game "
                    f"across {int(total_gp)} games. "
                    f"In the most recent season ({season_id}), {player_name} played {season_gp} games, "
                    f"averaging {season_ppg:.1f} points per game, {season_rpg:.1f} rebounds per game, "
                    f"{season_apg:.1f} assists per game, {season_tov_avg:.1f} turnovers per game, "
                    f"and {season_bpg:.1f} blocks per game."
                )
                sources.append(
                    {
                        "type": "player_career_stats",
                        "player_id": str(player_id),
                        "player_name": player_name,
                    }
                )

            # Recent games
            recent_games = player_data.get("recent_games", [])
            if recent_games:
                context_parts.append(f"\n{player_name}'s most recent games:")
                for game in recent_games[:3]:  # Last 3 games
                    game_date = game.get("GAME_DATE", "N/A")
                    matchup = game.get("MATCHUP", "N/A")
                    pts = game.get("PTS", 0)
                    reb = game.get("REB", 0)
                    ast = game.get("AST", 0)
                    context_parts.append(
                        f"On {game_date} against {matchup}, {player_name} scored {pts} points, "
                        f"{reb} rebounds, and {ast} assists."
                    )
                sources.append(
                    {
                        "type": "player_game_log",
                        "player_id": str(player_id),
                        "player_name": player_name,
                    }
                )

        return " ".join(context_parts), sources

    def _generate_game_context(
        self, data: Dict, question: str
    ) -> Tuple[str, List[Dict]]:
        """Generate context for game data"""
        context_parts = []
        sources = []

        for game_data in data.get("games", []):
            game_id = game_data.get("game_id", "N/A")
            game_date = game_data.get("game_date", "N/A")
            matchup = game_data.get("matchup", "N/A")

            # Check if we have direct score information (from last game query)
            team_score = game_data.get("team_score")
            opponent_score = game_data.get("opponent_score")
            result = game_data.get("result", "")

            if team_score is not None and opponent_score is not None:
                # Format: "On 2024-01-15, the Lakers played against the Warriors.
                # The Lakers won 120-115."
                # Use stored team_name and opponent_name if available, otherwise parse from matchup
                team_name = game_data.get("team_name")
                opponent_name = game_data.get("opponent_name")

                if not team_name:
                    # Fallback: parse from matchup
                    team_name = matchup.split()[0] if matchup else "Team"
                if not opponent_name:
                    # Fallback: parse from matchup
                    matchup_parts = matchup.split()
                    opponent_name = (
                        matchup_parts[-1] if len(matchup_parts) > 2 else "Opponent"
                    )

                context_parts.append(
                    f"On {game_date}, {team_name} played against {opponent_name}. "
                    f"{team_name} {'won' if result == 'W' else 'lost'} {team_score}-{opponent_score}."
                )

                sources.append(
                    {
                        "type": "team_game",
                        "game_id": game_id,
                        "game_date": game_date,
                        "team": team_name,
                    }
                )
            else:
                # Fallback to boxscore parsing
                boxscore = game_data.get("boxscore", [])
                if boxscore:
                    context_parts.append(f"Game {game_id} on {game_date}: {matchup}")

                    # Group by team
                    teams_data = {}
                    for player in boxscore:
                        team = player.get("TEAM_ABBREVIATION", "UNK")
                        if team not in teams_data:
                            teams_data[team] = {"players": [], "total_pts": 0}
                        teams_data[team]["players"].append(player)
                        teams_data[team]["total_pts"] += player.get("PTS", 0)

                    for team, team_info in teams_data.items():
                        context_parts.append(
                            f"{team} scored {team_info['total_pts']} points. "
                            f"Top performers: {', '.join([p.get('PLAYER_NAME', '') for p in team_info['players'][:3]])}"
                        )

                    sources.append(
                        {"type": "boxscore", "game_id": game_id, "game_date": game_date}
                    )

        return " ".join(context_parts), sources

    def _generate_comparison_context(
        self, data: Dict, question: str
    ) -> Tuple[str, List[Dict]]:
        """Generate context for player comparisons"""
        context_parts = []
        sources = []

        player_stats_list = []
        for player_data in data.get("players", []):
            player_name = player_data["player_name"]
            player_id = player_data["player_id"]
            career_stats = player_data.get("career_stats", [])

            if career_stats:
                # Calculate career averages for comparison
                total_pts = sum(season.get("PTS", 0) for season in career_stats)
                total_reb = sum(season.get("REB", 0) for season in career_stats)
                total_ast = sum(season.get("AST", 0) for season in career_stats)
                total_tov = sum(season.get("TOV", 0) for season in career_stats)
                total_blk = sum(season.get("BLK", 0) for season in career_stats)
                total_gp = sum(season.get("GP", 0) for season in career_stats)

                if total_gp > 0:
                    career_ppg = total_pts / total_gp
                    career_rpg = total_reb / total_gp
                    career_apg = total_ast / total_gp
                    career_tov = total_tov / total_gp
                    career_bpg = total_blk / total_gp
                else:
                    career_ppg = career_rpg = career_apg = career_tov = career_bpg = 0.0

                stats_summary = {
                    "name": player_name,
                    "ppg": career_ppg,
                    "rpg": career_rpg,
                    "apg": career_apg,
                    "tov": career_tov,
                    "bpg": career_bpg,
                    "games": int(total_gp),
                }
                player_stats_list.append(stats_summary)

                context_parts.append(
                    f"{player_name} averaged {stats_summary['ppg']:.1f} points, "
                    f"{stats_summary['rpg']:.1f} rebounds, {stats_summary['apg']:.1f} assists, "
                    f"{stats_summary['tov']:.1f} turnovers, and {stats_summary['bpg']:.1f} blocks "
                    f"in {stats_summary['games']} games."
                )

                sources.append(
                    {
                        "type": "player_career_stats",
                        "player_id": str(player_id),
                        "player_name": player_name,
                    }
                )

        return " ".join(context_parts), sources

    def _generate_play_by_play_context(
        self, data: Dict, question: str
    ) -> Tuple[str, List[Dict]]:
        """Generate context for play-by-play data"""
        context_parts = []
        sources = []

        game_id = data.get("game_id", "N/A")
        plays = data.get("plays", [])

        if plays:
            context_parts.append(f"Play-by-play for game {game_id}:")

            # Filter relevant plays based on question
            if "shot" in question.lower():
                shot_plays = [
                    p for p in plays if p.get("EVENTMSGTYPE") in [1, 2]
                ]  # Made/missed shots
                for play in shot_plays[:10]:  # First 10 shots
                    player = play.get("PLAYER1_NAME", "Unknown")
                    action = play.get("HOMEDESCRIPTION") or play.get(
                        "VISITORDESCRIPTION", ""
                    )
                    if action:
                        context_parts.append(f"{player}: {action}")
            else:
                # General play-by-play
                for play in plays[:20]:  # First 20 plays
                    player = play.get("PLAYER1_NAME", "")
                    action = play.get("HOMEDESCRIPTION") or play.get(
                        "VISITORDESCRIPTION", ""
                    )
                    if action and player:
                        context_parts.append(f"{player}: {action}")

            sources.append({"type": "play_by_play", "game_id": game_id})

        return " ".join(context_parts), sources

    def _generate_league_leaders_context(
        self, data: Dict, question: str
    ) -> Tuple[str, List[Dict]]:
        """Generate context for league leaders"""
        context_parts = []
        sources = []

        leaders = data.get("leaders", [])
        if not leaders:
            return "No league leaders data found.", []

        # Get stat type for context
        stat_type = leaders[0].get("stat_type", "statistics")
        stat_abbrev = leaders[0].get("stat_abbrev", "PTS")

        # Format stat name for display
        stat_names = {
            "points": "points",
            "rebounds": "rebounds",
            "assists": "assists",
            "steals": "steals",
            "blocks": "blocks",
            "turnovers": "turnovers",
            "three_pointers": "three-pointers",
        }
        stat_display = stat_names.get(stat_type, "statistics")

        context_parts.append(
            f"The top {len(leaders)} players with the most {stat_display} in NBA history are:"
        )

        for leader in leaders:
            rank = leader.get("rank", 0)
            player_name = leader.get("player_name", "Unknown")
            stat_value = leader.get("stat_value", 0)

            # Format stat value (integer for totals, decimal for percentages)
            if stat_abbrev in ["FG_PCT", "FG3_PCT", "FT_PCT"]:
                stat_display_value = f"{stat_value:.3f}"
            else:
                stat_display_value = f"{int(stat_value):,}"

            context_parts.append(
                f"{rank}. {player_name} with {stat_display_value} {stat_display}."
            )

        sources.append(
            {
                "type": "league_leaders",
                "stat_type": stat_type,
                "stat_abbrev": stat_abbrev,
            }
        )

        return " ".join(context_parts), sources


class NBAQASystem:
    """Main QA system integrating NBA API with Hugging Face QA model"""

    def __init__(self, model_name: str = "deepset/roberta-base-squad2"):
        """Initialize the QA system"""
        self.qa_pipeline = pipeline("question-answering", model=model_name)
        self.analyzer = QuestionAnalyzer()
        self.retriever = NBADataRetriever()
        self.context_generator = ContextGenerator()

    def answer(self, question: str) -> QAAnswer:
        """
        Main method to answer a question about NBA

        Args:
            question: Natural language question about NBA

        Returns:
            QAAnswer object with answer, confidence, context, and sources
        """
        # Step 1: Analyze question
        analysis = self.analyzer.analyze(question)

        # Step 2: Retrieve NBA data
        nba_data = self.retriever.retrieve(analysis)

        # Step 3: Generate context
        context, sources = self.context_generator.generate(nba_data, question)

        if (
            not context
            or context == "No specific NBA data found to answer this question."
        ):
            return QAAnswer(
                answer="I couldn't find relevant NBA data to answer this question. Please try asking about specific players, teams, or games.",
                confidence=0.0,
                context_used="",
                sources=[],
                raw_data=nba_data,
            )

        # Step 4: Special handling for "top N" league leader questions
        # These need to return a full list, not a single extracted answer
        if (
            analysis.question_type == "league_leaders"
            and analysis.entities.get("top_n") is not None
            and nba_data.get("type") == "league_leaders"
        ):
            leaders = nba_data.get("leaders", [])
            if leaders:
                # Get stat abbreviation for column header
                stat_abbrev = leaders[0].get("stat_abbrev", "STAT")

                # Format as a table-like list
                answer_lines = []
                # Header
                answer_lines.append(f"{'Rank':<6} {'Player Name':<20} {stat_abbrev}")
                answer_lines.append("-" * 50)

                for leader in leaders:
                    rank = leader.get("rank", 0)
                    player_name = leader.get("player_name", "Unknown")
                    stat_value = leader.get("stat_value", 0)

                    # Format stat value
                    if stat_abbrev in ["FG_PCT", "FG3_PCT", "FT_PCT"]:
                        stat_display = f"{stat_value:.3f}"
                    else:
                        stat_display = f"{int(stat_value):,}"

                    answer_lines.append(f"{rank:<6} {player_name:<20} {stat_display}")

                answer = "\n".join(answer_lines)
                confidence = 0.95  # High confidence for structured data
            else:
                answer = "No league leaders data found."
                confidence = 0.0
        # Special handling for "last game" questions - format score directly
        elif (
            analysis.question_type == "game_result"
            and analysis.entities.get("temporal", {}).get("last", False)
            and nba_data.get("type") == "game_data"
        ):
            games = nba_data.get("games", [])
            if games and len(games) > 0:
                game = games[0]
                team_score = game.get("team_score")
                opponent_score = game.get("opponent_score")
                team_name = game.get("team_name", "")
                opponent_name = game.get("opponent_name", "")
                result = game.get("result", "")

                if team_score is not None and opponent_score is not None:
                    # Format: "Lakers 120, Warriors 115" or "Lakers won 120-115"
                    if "score" in question.lower():
                        answer = f"{team_name} {team_score}, {opponent_name} {opponent_score}"
                    else:
                        answer = f"{team_name} {'won' if result == 'W' else 'lost'} {team_score}-{opponent_score}"
                    confidence = 0.95
                else:
                    # Fallback to QA model
                    try:
                        qa_result = self.qa_pipeline(question=question, context=context)
                        answer = qa_result["answer"]
                        confidence = qa_result["score"]
                    except Exception as e:
                        print(f"Error in QA model: {e}")
                        answer = (
                            context[:200] + "..." if len(context) > 200 else context
                        )
                        confidence = 0.5
            else:
                # Fallback to QA model
                try:
                    qa_result = self.qa_pipeline(question=question, context=context)
                    answer = qa_result["answer"]
                    confidence = qa_result["score"]
                except Exception as e:
                    print(f"Error in QA model: {e}")
                    answer = context[:200] + "..." if len(context) > 200 else context
                    confidence = 0.5
        else:
            # Step 5: Use QA model for single-answer questions
            try:
                qa_result = self.qa_pipeline(question=question, context=context)
                answer = qa_result["answer"]
                confidence = qa_result["score"]
            except Exception as e:
                print(f"Error in QA model: {e}")
                # Fallback: return context if QA fails
                answer = context[:200] + "..." if len(context) > 200 else context
                confidence = 0.5

        # Step 5: Format answer
        return QAAnswer(
            answer=answer,
            confidence=confidence,
            context_used=context,
            sources=sources,
            raw_data=nba_data,
        )

    def answer_with_details(self, question: str) -> Dict[str, Any]:
        """Answer question with full details for debugging"""
        analysis = self.analyzer.analyze(question)
        nba_data = self.retriever.retrieve(analysis)
        context, sources = self.context_generator.generate(nba_data, question)

        qa_result = None
        if context:
            try:
                qa_result = self.qa_pipeline(question=question, context=context)
            except Exception as e:
                print(f"Error in QA model: {e}")

        return {
            "question": question,
            "analysis": {
                "question_type": analysis.question_type,
                "entities": analysis.entities,
                "confidence": analysis.confidence,
            },
            "nba_data_type": nba_data.get("type"),
            "context": context,
            "sources": sources,
            "qa_result": qa_result,
            "answer": qa_result["answer"] if qa_result else "Could not generate answer",
            "confidence": qa_result["score"] if qa_result else 0.0,
        }


# Example usage
if __name__ == "__main__":
    # Initialize the system
    qa_system = NBAQASystem()

    # Example questions
    example_questions = [
        "What is LeBron James' career scoring average?",
        "How many points did Stephen Curry score in his last game?",
        "What was the score of the last Lakers game?",
        "Who scored more points, LeBron James or Kevin Durant?",
        "Who are the top 5 assists leaders of the league history?",
        "What was the Kobe Bryant's career PPG?",
        "What was the score of the last Lakers game?",
    ]

    print("NBA Question Answering System - Example Queries\n")
    print("=" * 60)

    for question in example_questions:
        print(f"\nQuestion: {question}")
        print("-" * 60)

        answer = qa_system.answer(question)

        print(f"Answer: {answer.answer}")
        print(f"Confidence: {answer.confidence:.2f}")
        print(f"Sources: {len(answer.sources)} source(s)")
        for source in answer.sources:
            print(f"  - {source.get('type', 'unknown')}: {source}")

        print()
