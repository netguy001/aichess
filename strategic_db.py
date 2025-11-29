import json
import os
from datetime import datetime
from collections import defaultdict
import hashlib


class StrategicDatabase:
    """
    Advanced strategic learning database for chess AI.
    Stores and retrieves strategic patterns, successful tactics, and game wisdom.
    """

    def __init__(self, db_path="data/strategic_brain.json"):
        self.db_path = db_path
        self.data = self._load_or_create_db()

    def _load_or_create_db(self):
        """Load existing database or create new one"""
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, "r") as f:
                    return json.load(f)
            except:
                return self._create_empty_db()
        else:
            return self._create_empty_db()

    def _create_empty_db(self):
        """Create empty database structure"""
        return {
            # Opening book: position_hash -> {move: {wins, losses, draws, avg_score}}
            "opening_book": {},
            # Middle game patterns: position_type -> {strategy: success_rate}
            "middlegame_strategies": {
                "open_position": {},
                "closed_position": {},
                "tactical_position": {},
                "strategic_position": {},
            },
            # Endgame knowledge: piece_configuration -> best_strategy
            "endgame_knowledge": {},
            # Tactical patterns that worked
            "successful_tactics": {
                "forks": [],  # Position hashes where forks won
                "pins": [],
                "skewers": [],
                "discovered_attacks": [],
                "sacrifices": [],
            },
            # Defensive patterns that prevented loss
            "defensive_patterns": {
                "escape_sequences": [],  # How to escape bad positions
                "counter_attacks": [],  # Successful counter-attack positions
                "fortress_positions": [],  # Defensive setups that held
            },
            # Position evaluations: position_hash -> {eval, depth, games_seen}
            "position_memory": {},
            # Move sequences: hash -> {sequence: [moves], outcome, frequency}
            "winning_sequences": {},
            "losing_sequences": {},
            # Opponent pattern recognition
            "opponent_patterns": {
                "common_mistakes": [],  # Player mistakes we can exploit
                "strong_defenses": [],  # Player's good defenses to avoid
            },
            # Strategic concepts learned
            "strategic_concepts": {
                "piece_coordination": 0.5,  # How important is coordination
                "center_control": 0.7,
                "king_safety": 0.8,
                "pawn_structure": 0.6,
                "piece_activity": 0.7,
            },
            # Statistics
            "stats": {
                "total_positions_analyzed": 0,
                "patterns_learned": 0,
                "successful_predictions": 0,
                "failed_predictions": 0,
                "last_updated": None,
            },
        }

    def save(self):
        """Save database to disk"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.data["stats"]["last_updated"] = datetime.now().isoformat()
        with open(self.db_path, "w") as f:
            json.dump(self.data, f, indent=2)

    # ===== OPENING BOOK METHODS =====

    def record_opening_move(self, position_hash, move_str, outcome, score):
        """
        Record an opening move and its outcome
        outcome: 'win', 'loss', 'draw'
        """
        if position_hash not in self.data["opening_book"]:
            self.data["opening_book"][position_hash] = {}

        if move_str not in self.data["opening_book"][position_hash]:
            self.data["opening_book"][position_hash][move_str] = {
                "wins": 0,
                "losses": 0,
                "draws": 0,
                "total_games": 0,
                "avg_score": 0,
                "scores": [],
            }

        move_data = self.data["opening_book"][position_hash][move_str]
        move_data["total_games"] += 1

        if outcome == "win":
            move_data["wins"] += 1
        elif outcome == "loss":
            move_data["losses"] += 1
        else:
            move_data["draws"] += 1

        # Update average score
        move_data["scores"].append(score)
        if len(move_data["scores"]) > 100:  # Keep only last 100 scores
            move_data["scores"] = move_data["scores"][-100:]
        move_data["avg_score"] = sum(move_data["scores"]) / len(move_data["scores"])

        self.data["stats"]["patterns_learned"] += 1

    def get_opening_book_move(self, position_hash):
        """
        Get best opening move from book based on success rate
        Returns: move_str or None
        """
        if position_hash not in self.data["opening_book"]:
            return None

        moves = self.data["opening_book"][position_hash]
        if not moves:
            return None

        # Calculate success rate for each move
        best_move = None
        best_score = -float("inf")

        for move_str, data in moves.items():
            if data["total_games"] < 3:  # Need at least 3 games
                continue

            # Success rate = (wins + 0.5*draws) / total_games
            success_rate = (data["wins"] + 0.5 * data["draws"]) / data["total_games"]
            # Combine with average position score
            combined_score = success_rate * 0.7 + (data["avg_score"] / 1000) * 0.3

            if combined_score > best_score:
                best_score = combined_score
                best_move = move_str

        return best_move if best_score > 0.4 else None  # Only use if decent success

    # ===== TACTICAL PATTERN METHODS =====

    def record_successful_tactic(self, tactic_type, position_hash, move_sequence):
        """Record a successful tactical pattern"""
        if tactic_type in self.data["successful_tactics"]:
            self.data["successful_tactics"][tactic_type].append(
                {
                    "position": position_hash,
                    "moves": move_sequence,
                    "timestamp": datetime.now().isoformat(),
                }
            )
            # Keep only last 200 patterns
            if len(self.data["successful_tactics"][tactic_type]) > 200:
                self.data["successful_tactics"][tactic_type] = self.data[
                    "successful_tactics"
                ][tactic_type][-200:]

    def has_similar_tactical_pattern(self, tactic_type, position_hash):
        """Check if we've seen similar tactical pattern before"""
        if tactic_type not in self.data["successful_tactics"]:
            return False

        patterns = self.data["successful_tactics"][tactic_type]
        return any(p["position"] == position_hash for p in patterns[-50:])

    # ===== STRATEGIC CONCEPT METHODS =====

    def update_strategic_weight(self, concept, success):
        """
        Update the importance weight of a strategic concept
        concept: 'center_control', 'king_safety', etc.
        success: True/False
        """
        if concept in self.data["strategic_concepts"]:
            current = self.data["strategic_concepts"][concept]
            # Gradual learning: adjust by 0.05
            if success:
                self.data["strategic_concepts"][concept] = min(1.0, current + 0.05)
            else:
                self.data["strategic_concepts"][concept] = max(0.1, current - 0.03)

    def get_strategic_weight(self, concept):
        """Get importance weight of a strategic concept"""
        return self.data["strategic_concepts"].get(concept, 0.5)

    # ===== POSITION MEMORY METHODS =====

    def remember_position(self, position_hash, evaluation, depth):
        """Store position evaluation for future reference"""
        if position_hash not in self.data["position_memory"]:
            self.data["position_memory"][position_hash] = {
                "eval": evaluation,
                "depth": depth,
                "games_seen": 1,
                "last_seen": datetime.now().isoformat(),
            }
        else:
            mem = self.data["position_memory"][position_hash]
            # Update with weighted average
            mem["games_seen"] += 1
            mem["eval"] = (mem["eval"] * 0.7) + (evaluation * 0.3)
            mem["depth"] = max(mem["depth"], depth)
            mem["last_seen"] = datetime.now().isoformat()

        self.data["stats"]["total_positions_analyzed"] += 1

        # Limit size: keep only most recently seen positions
        if len(self.data["position_memory"]) > 10000:
            # Remove oldest 1000 positions
            sorted_positions = sorted(
                self.data["position_memory"].items(),
                key=lambda x: x[1]["last_seen"],
            )
            for pos_hash, _ in sorted_positions[:1000]:
                del self.data["position_memory"][pos_hash]

    def recall_position(self, position_hash):
        """Retrieve remembered position evaluation"""
        return self.data["position_memory"].get(position_hash, None)

    # ===== SEQUENCE METHODS =====

    def record_winning_sequence(self, move_sequence, final_score):
        """Record a sequence of moves that led to victory"""
        seq_hash = hashlib.md5(str(move_sequence).encode()).hexdigest()

        if seq_hash not in self.data["winning_sequences"]:
            self.data["winning_sequences"][seq_hash] = {
                "sequence": move_sequence,
                "frequency": 1,
                "avg_score": final_score,
                "scores": [final_score],
            }
        else:
            seq = self.data["winning_sequences"][seq_hash]
            seq["frequency"] += 1
            seq["scores"].append(final_score)
            if len(seq["scores"]) > 50:
                seq["scores"] = seq["scores"][-50:]
            seq["avg_score"] = sum(seq["scores"]) / len(seq["scores"])

        # Keep only top 500 sequences
        if len(self.data["winning_sequences"]) > 500:
            sorted_seqs = sorted(
                self.data["winning_sequences"].items(),
                key=lambda x: x[1]["frequency"],
                reverse=True,
            )
            self.data["winning_sequences"] = dict(sorted_seqs[:500])

    def record_losing_sequence(self, move_sequence):
        """Record a sequence that led to defeat (to avoid)"""
        seq_hash = hashlib.md5(str(move_sequence).encode()).hexdigest()

        if seq_hash not in self.data["losing_sequences"]:
            self.data["losing_sequences"][seq_hash] = {
                "sequence": move_sequence,
                "frequency": 1,
            }
        else:
            self.data["losing_sequences"][seq_hash]["frequency"] += 1

        # Keep only most frequent 300
        if len(self.data["losing_sequences"]) > 300:
            sorted_seqs = sorted(
                self.data["losing_sequences"].items(),
                key=lambda x: x[1]["frequency"],
                reverse=True,
            )
            self.data["losing_sequences"] = dict(sorted_seqs[:300])

    def is_losing_pattern(self, move_sequence):
        """Check if move sequence matches a known losing pattern"""
        seq_hash = hashlib.md5(str(move_sequence).encode()).hexdigest()
        return seq_hash in self.data["losing_sequences"]

    # ===== DEFENSIVE PATTERN METHODS =====

    def record_escape_sequence(self, from_position, escape_moves, success):
        """Record how we escaped a bad position"""
        if success:
            self.data["defensive_patterns"]["escape_sequences"].append(
                {
                    "position": from_position,
                    "moves": escape_moves,
                    "timestamp": datetime.now().isoformat(),
                }
            )
            # Keep last 100
            if len(self.data["defensive_patterns"]["escape_sequences"]) > 100:
                self.data["defensive_patterns"]["escape_sequences"] = self.data[
                    "defensive_patterns"
                ]["escape_sequences"][-100:]

    def find_escape_pattern(self, current_position):
        """Find if we know how to escape current bad position"""
        for pattern in self.data["defensive_patterns"]["escape_sequences"][-20:]:
            if pattern["position"] == current_position:
                return pattern["moves"]
        return None

    # ===== STATISTICS AND ANALYSIS =====

    def get_learning_stats(self):
        """Get statistics about what the AI has learned"""
        return {
            "opening_positions": len(self.data["opening_book"]),
            "positions_remembered": len(self.data["position_memory"]),
            "winning_patterns": len(self.data["winning_sequences"]),
            "losing_patterns": len(self.data["losing_sequences"]),
            "tactical_patterns": sum(
                len(v) for v in self.data["successful_tactics"].values()
            ),
            "total_analyzed": self.data["stats"]["total_positions_analyzed"],
            "patterns_learned": self.data["stats"]["patterns_learned"],
            "strategic_weights": self.data["strategic_concepts"],
        }

    def reset_database(self):
        """Reset all learning (use carefully!)"""
        self.data = self._create_empty_db()
        self.save()
