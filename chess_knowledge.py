"""
Chess Knowledge Base - Hard-coded chess wisdom
This module contains expert chess knowledge that makes the AI strong immediately.
"""


class ChessKnowledge:
    """Expert chess knowledge system"""
    
    def __init__(self):
        # Opening principles (first 10-15 moves)
        self.opening_principles = {
            "control_center": {
                "squares": [(3, 3), (3, 4), (4, 3), (4, 4)],
                "importance": 0.9,
                "bonus_per_piece": 50
            },
            "develop_minor_pieces": {
                "ideal_squares": {
                    "knight": [(2, 2), (2, 5), (5, 2), (5, 5)],  # Black knights
                    "bishop": [(2, 3), (2, 4), (3, 2), (3, 5), (4, 2), (4, 5)]
                },
                "bonus": 40
            },
            "castle_early": {
                "move_range": (5, 12),
                "bonus": 150
            },
            "dont_move_same_piece_twice": {
                "penalty": -30
            },
            "connect_rooks": {
                "bonus": 80
            }
        }
        
        # Tactical pattern library
        self.tactical_patterns = {
            "back_rank_mate": {
                "description": "King trapped on back rank",
                "detection_bonus": 300,
                "setup_bonus": 150
            },
            "smothered_mate": {
                "description": "King trapped by own pieces, knight delivers mate",
                "detection_bonus": 400,
                "setup_bonus": 200
            },
            "queen_knight_battery": {
                "description": "Queen and knight attacking together",
                "bonus": 120
            },
            "rook_on_seventh": {
                "description": "Rook on opponent's 7th rank",
                "bonus": 100
            },
            "bishop_pair": {
                "description": "Both bishops vs opponent without",
                "bonus": 50
            }
        }
        
        # Endgame knowledge
        self.endgame_tablebase = {
            "KQvK": {
                "description": "King + Queen vs King",
                "method": "drive_king_to_edge",
                "winning": True
            },
            "KRvK": {
                "description": "King + Rook vs King",
                "method": "drive_king_to_edge",
                "winning": True
            },
            "KBBvK": {
                "description": "King + 2 Bishops vs King",
                "method": "checkmate_possible",
                "winning": True
            },
            "KBNvK": {
                "description": "King + Bishop + Knight vs King",
                "method": "corner_checkmate",
                "winning": True
            },
            "KBvK": {
                "description": "King + Bishop vs King",
                "method": "insufficient_material",
                "winning": False
            },
            "KNvK": {
                "description": "King + Knight vs King",
                "method": "insufficient_material",
                "winning": False
            }
        }
        
        # Positional concepts
        self.positional_concepts = {
            "outpost": {
                "description": "Protected piece in enemy territory",
                "bonus": 60
            },
            "weak_squares": {
                "description": "Squares that can't be defended by pawns",
                "penalty": -40
            },
            "pawn_majority": {
                "description": "More pawns on one side of the board",
                "bonus": 35
            },
            "bad_bishop": {
                "description": "Bishop blocked by own pawns",
                "penalty": -50
            },
            "rook_behind_passed_pawn": {
                "description": "Rook supporting passed pawn from behind",
                "bonus": 80
            }
        }
    
    def evaluate_opening_principles(self, board, color, move_count):
        """
        Evaluate how well opening principles are being followed
        Returns bonus/penalty score
        """
        if move_count > 15:
            return 0  # Not in opening anymore
        
        bonus = 0
        
        # Check center control
        center_control = self._check_center_control(board, color)
        bonus += center_control
        
        # Check piece development
        development = self._check_piece_development(board, color, move_count)
        bonus += development
        
        # Check castling status
        castling_bonus = self._check_castling(board, color, move_count)
        bonus += castling_bonus
        
        # Check if rooks are connected
        if move_count > 8:
            rook_connection = self._check_rook_connection(board, color)
            bonus += rook_connection
        
        return bonus
    
    def _check_center_control(self, board, color):
        """Check control of center squares"""
        bonus = 0
        center_squares = self.opening_principles["control_center"]["squares"]
        
        for r, c in center_squares:
            piece = board[r][c]
            if piece:
                is_our_piece = (color == "black" and piece.islower()) or \
                               (color == "white" and piece.isupper())
                if is_our_piece:
                    bonus += self.opening_principles["control_center"]["bonus_per_piece"]
                else:
                    bonus -= self.opening_principles["control_center"]["bonus_per_piece"] * 0.5
        
        return bonus
    
    def _check_piece_development(self, board, color, move_count):
        """Check if minor pieces are developed"""
        bonus = 0
        
        if color == "black":
            back_rank = 0
            piece_chars = {'n', 'b'}
            developed_squares = [(2, 2), (2, 5), (5, 2), (5, 5)]  # Knights
            developed_squares.extend([(2, 3), (2, 4), (3, 2), (3, 5)])  # Bishops
        else:
            back_rank = 7
            piece_chars = {'N', 'B'}
            developed_squares = [(5, 2), (5, 5), (6, 2), (6, 5)]  # Knights
            developed_squares.extend([(5, 3), (5, 4), (4, 2), (4, 5)])  # Bishops
        
        # Penalty for pieces still on back rank
        pieces_on_back_rank = 0
        for c in [1, 2, 5, 6]:  # Knight and bishop starting squares
            piece = board[back_rank][c]
            if piece and piece.lower() in piece_chars:
                pieces_on_back_rank += 1
        
        # After move 8, penalize undeveloped pieces heavily
        if move_count > 8:
            bonus -= pieces_on_back_rank * 60
        else:
            bonus -= pieces_on_back_rank * 30
        
        # Bonus for developed pieces
        for r, c in developed_squares:
            piece = board[r][c]
            if piece and piece.lower() in piece_chars:
                is_our_piece = (color == "black" and piece.islower()) or \
                               (color == "white" and piece.isupper())
                if is_our_piece:
                    bonus += 35
        
        return bonus
    
    def _check_castling(self, board, color, move_count):
        """Check if castled or can castle"""
        # This would need castling rights info from the main AI
        # For now, check if king is on starting square
        bonus = 0
        
        if color == "black":
            king_start = (0, 4)
            castled_positions = [(0, 2), (0, 6)]
        else:
            king_start = (7, 4)
            castled_positions = [(7, 2), (7, 6)]
        
        # Find king
        king_char = 'k' if color == "black" else 'K'
        king_pos = None
        
        for r in range(8):
            for c in range(8):
                if board[r][c] == king_char:
                    king_pos = (r, c)
                    break
            if king_pos:
                break
        
        if king_pos:
            if king_pos in castled_positions:
                # Already castled - great!
                bonus += 150
            elif king_pos == king_start and move_count > 10:
                # Haven't castled by move 10 - risky
                bonus -= 100
        
        return bonus
    
    def _check_rook_connection(self, board, color):
        """Check if rooks are connected (on same rank with nothing between)"""
        rook_char = 'r' if color == "black" else 'R'
        rooks = []
        
        for r in range(8):
            for c in range(8):
                if board[r][c] == rook_char:
                    rooks.append((r, c))
        
        if len(rooks) == 2:
            r1, c1 = rooks[0]
            r2, c2 = rooks[1]
            
            # Check if on same rank
            if r1 == r2:
                # Check if path is clear
                min_col = min(c1, c2)
                max_col = max(c1, c2)
                
                clear = True
                for c in range(min_col + 1, max_col):
                    if board[r1][c]:
                        clear = False
                        break
                
                if clear:
                    return self.opening_principles["connect_rooks"]["bonus"]
        
        return 0
    
    def detect_tactical_pattern(self, board, color):
        """
        Detect tactical patterns in current position
        Returns: bonus score and pattern description
        """
        total_bonus = 0
        patterns_found = []
        
        # Check for back rank mate threat
        back_rank_bonus, back_rank_desc = self._check_back_rank_weakness(board, color)
        if back_rank_bonus > 0:
            total_bonus += back_rank_bonus
            patterns_found.append(back_rank_desc)
        
        # Check for rook on 7th rank
        rook_7th_bonus = self._check_rook_on_seventh(board, color)
        if rook_7th_bonus > 0:
            total_bonus += rook_7th_bonus
            patterns_found.append("Rook on 7th rank")
        
        # Check for bishop pair advantage
        bishop_pair_bonus = self._check_bishop_pair(board, color)
        if bishop_pair_bonus > 0:
            total_bonus += bishop_pair_bonus
            patterns_found.append("Bishop pair advantage")
        
        # Check for queen-knight battery
        qn_battery_bonus = self._check_queen_knight_battery(board, color)
        if qn_battery_bonus > 0:
            total_bonus += qn_battery_bonus
            patterns_found.append("Queen-knight battery")
        
        return total_bonus, patterns_found
    
    def _check_back_rank_weakness(self, board, color):
        """Check if opponent has back rank mate vulnerability"""
        opponent_color = "white" if color == "black" else "black"
        opponent_king_char = 'K' if opponent_color == "white" else 'k'
        opponent_back_rank = 7 if opponent_color == "white" else 0
        
        # Find opponent's king
        king_col = None
        for c in range(8):
            if board[opponent_back_rank][c] == opponent_king_char:
                king_col = c
                break
        
        if king_col is None:
            return 0, ""
        
        # Check if king is trapped on back rank
        # (pawns in front, no escape squares)
        pawn_char = 'P' if opponent_color == "white" else 'p'
        front_rank = opponent_back_rank - 1 if opponent_color == "white" else opponent_back_rank + 1
        
        # Count pawns blocking king
        blocking_pawns = 0
        for dc in [-1, 0, 1]:
            check_col = king_col + dc
            if 0 <= check_col < 8 and 0 <= front_rank < 8:
                if board[front_rank][check_col] == pawn_char:
                    blocking_pawns += 1
        
        if blocking_pawns >= 2:
            # King is trapped! Check if we have attacking pieces
            our_rook = 'r' if color == "black" else 'R'
            our_queen = 'q' if color == "black" else 'Q'
            
            # Check if we have rook or queen that can attack back rank
            for c in range(8):
                piece = board[opponent_back_rank][c]
                if piece in [our_rook, our_queen]:
                    return self.tactical_patterns["back_rank_mate"]["detection_bonus"], \
                           "Back rank mate threat!"
            
            # Potential setup
            return self.tactical_patterns["back_rank_mate"]["setup_bonus"], \
                   "Back rank weakness"
        
        return 0, ""
    
    def _check_rook_on_seventh(self, board, color):
        """Check if rook is on opponent's 7th rank"""
        rook_char = 'r' if color == "black" else 'R'
        seventh_rank = 6 if color == "black" else 1
        
        for c in range(8):
            if board[seventh_rank][c] == rook_char:
                return self.tactical_patterns["rook_on_seventh"]["bonus"]
        
        return 0
    
    def _check_bishop_pair(self, board, color):
        """Check if we have both bishops and opponent doesn't"""
        our_bishops = 0
        opponent_bishops = 0
        
        for r in range(8):
            for c in range(8):
                piece = board[r][c]
                if piece and piece.lower() == 'b':
                    if (color == "black" and piece.islower()) or \
                       (color == "white" and piece.isupper()):
                        our_bishops += 1
                    else:
                        opponent_bishops += 1
        
        if our_bishops == 2 and opponent_bishops < 2:
            return self.tactical_patterns["bishop_pair"]["bonus"]
        
        return 0
    
    def _check_queen_knight_battery(self, board, color):
        """Check for queen and knight coordinating attacks"""
        queen_char = 'q' if color == "black" else 'Q'
        knight_char = 'n' if color == "black" else 'N'
        
        # Find queen and knights
        queen_pos = None
        knight_positions = []
        
        for r in range(8):
            for c in range(8):
                piece = board[r][c]
                if piece == queen_char:
                    queen_pos = (r, c)
                elif piece == knight_char:
                    knight_positions.append((r, c))
        
        if not queen_pos or not knight_positions:
            return 0
        
        # Check if queen and knight are attacking common squares
        # (simplified check - just proximity)
        qr, qc = queen_pos
        for kr, kc in knight_positions:
            distance = abs(qr - kr) + abs(qc - kc)
            if distance <= 3:  # Close together
                return self.tactical_patterns["queen_knight_battery"]["bonus"]
        
        return 0
    
    def get_endgame_strategy(self, board):
        """
        Identify endgame type and return optimal strategy
        """
        material = self._count_material(board)
        
        # Identify endgame type
        endgame_type = self._identify_endgame_type(material)
        
        if endgame_type in self.endgame_tablebase:
            return self.endgame_tablebase[endgame_type]
        
        return None
    
    def _count_material(self, board):
        """Count remaining material"""
        material = {
            'white': {'K': 0, 'Q': 0, 'R': 0, 'B': 0, 'N': 0, 'P': 0},
            'black': {'k': 0, 'q': 0, 'r': 0, 'b': 0, 'n': 0, 'p': 0}
        }
        
        for r in range(8):
            for c in range(8):
                piece = board[r][c]
                if piece:
                    if piece.isupper():
                        material['white'][piece] += 1
                    else:
                        material['black'][piece] += 1
        
        return material
    
    def _identify_endgame_type(self, material):
        """Identify specific endgame type"""
        white = material['white']
        black = material['black']
        
        # King + Queen vs King
        if white['Q'] == 1 and sum(white.values()) == 2 and sum(black.values()) == 1:
            return "KQvK"
        if black['q'] == 1 and sum(black.values()) == 2 and sum(white.values()) == 1:
            return "KQvK"
        
        # King + Rook vs King
        if white['R'] == 1 and sum(white.values()) == 2 and sum(black.values()) == 1:
            return "KRvK"
        if black['r'] == 1 and sum(black.values()) == 2 and sum(white.values()) == 1:
            return "KRvK"
        
        # King + 2 Bishops vs King
        if white['B'] == 2 and sum(white.values()) == 3 and sum(black.values()) == 1:
            return "KBBvK"
        if black['b'] == 2 and sum(black.values()) == 3 and sum(white.values()) == 1:
            return "KBBvK"
        
        # King + Bishop + Knight vs King
        if white['B'] == 1 and white['N'] == 1 and sum(white.values()) == 3 and sum(black.values()) == 1:
            return "KBNvK"
        if black['b'] == 1 and black['n'] == 1 and sum(black.values()) == 3 and sum(white.values()) == 1:
            return "KBNvK"
        
        return "unknown"
    
    def suggest_opening_move(self, board, move_count, color):
        """
        Suggest a strong opening move based on principles
        Returns: bonus scores for different move types
        """
        suggestions = {
            "center_pawn_move": 0,
            "knight_development": 0,
            "bishop_development": 0,
            "castling": 0,
            "queen_early": 0  # Negative - don't develop queen too early!
        }
        
        if move_count <= 3:
            suggestions["center_pawn_move"] = 80
            suggestions["knight_development"] = 60
            suggestions["queen_early"] = -100
        elif move_count <= 8:
            suggestions["knight_development"] = 70
            suggestions["bishop_development"] = 70
            suggestions["castling"] = 100
            suggestions["queen_early"] = -60
        elif move_count <= 12:
            suggestions["castling"] = 120
            suggestions["bishop_development"] = 50
            suggestions["center_pawn_move"] = 40
        
        return suggestions
