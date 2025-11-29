import random
import copy
import json
import hashlib
import time
from datetime import datetime
from collections import defaultdict
from strategic_db import StrategicDatabase


class ChessAI:
    def __init__(self):
        # Initialize strategic database
        self.strategic_db = StrategicDatabase()

        # Enhanced piece values
        self.piece_values = {
            "p": 100,
            "n": 320,
            "b": 330,
            "r": 500,
            "q": 900,
            "k": 20000,
            "P": -100,
            "N": -320,
            "B": -330,
            "R": -500,
            "Q": -900,
            "K": -20000,
        }

        # Position bonus tables (from black's perspective)
        self.pawn_table = [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [50, 50, 50, 50, 50, 50, 50, 50],
            [10, 10, 20, 30, 30, 20, 10, 10],
            [5, 5, 10, 27, 27, 10, 5, 5],
            [0, 0, 0, 25, 25, 0, 0, 0],
            [5, -5, -10, 0, 0, -10, -5, 5],
            [5, 10, 10, -25, -25, 10, 10, 5],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ]

        self.knight_table = [
            [-50, -40, -30, -30, -30, -30, -40, -50],
            [-40, -20, 0, 0, 0, 0, -20, -40],
            [-30, 0, 10, 15, 15, 10, 0, -30],
            [-30, 5, 15, 20, 20, 15, 5, -30],
            [-30, 0, 15, 20, 20, 15, 0, -30],
            [-30, 5, 10, 15, 15, 10, 5, -30],
            [-40, -20, 0, 5, 5, 0, -20, -40],
            [-50, -40, -30, -30, -30, -30, -40, -50],
        ]

        self.bishop_table = [
            [-20, -10, -10, -10, -10, -10, -10, -20],
            [-10, 0, 0, 0, 0, 0, 0, -10],
            [-10, 0, 5, 10, 10, 5, 0, -10],
            [-10, 5, 5, 10, 10, 5, 5, -10],
            [-10, 0, 10, 10, 10, 10, 0, -10],
            [-10, 10, 10, 10, 10, 10, 10, -10],
            [-10, 5, 0, 0, 0, 0, 5, -10],
            [-20, -10, -10, -10, -10, -10, -10, -20],
        ]

        self.rook_table = [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [5, 10, 10, 10, 10, 10, 10, 5],
            [-5, 0, 0, 0, 0, 0, 0, -5],
            [-5, 0, 0, 0, 0, 0, 0, -5],
            [-5, 0, 0, 0, 0, 0, 0, -5],
            [-5, 0, 0, 0, 0, 0, 0, -5],
            [-5, 0, 0, 0, 0, 0, 0, -5],
            [0, 0, 0, 5, 5, 0, 0, 0],
        ]

        self.queen_table = [
            [-20, -10, -10, -5, -5, -10, -10, -20],
            [-10, 0, 0, 0, 0, 0, 0, -10],
            [-10, 0, 5, 5, 5, 5, 0, -10],
            [-5, 0, 5, 5, 5, 5, 0, -5],
            [0, 0, 5, 5, 5, 5, 0, -5],
            [-10, 5, 5, 5, 5, 5, 0, -10],
            [-10, 0, 5, 0, 0, 0, 0, -10],
            [-20, -10, -10, -5, -5, -10, -10, -20],
        ]

        self.king_middle_table = [
            [-30, -40, -40, -50, -50, -40, -40, -30],
            [-30, -40, -40, -50, -50, -40, -40, -30],
            [-30, -40, -40, -50, -50, -40, -40, -30],
            [-30, -40, -40, -50, -50, -40, -40, -30],
            [-20, -30, -30, -40, -40, -30, -30, -20],
            [-10, -20, -20, -20, -20, -20, -20, -10],
            [20, 20, 0, 0, 0, 0, 20, 20],
            [20, 30, 10, 0, 0, 10, 30, 20],
        ]

        self.king_end_table = [
            [-50, -40, -30, -20, -20, -30, -40, -50],
            [-30, -20, -10, 0, 0, -10, -20, -30],
            [-30, -10, 20, 30, 30, 20, -10, -30],
            [-30, -10, 30, 40, 40, 30, -10, -30],
            [-30, -10, 30, 40, 40, 30, -10, -30],
            [-30, -10, 20, 30, 30, 20, -10, -30],
            [-30, -30, 0, 0, 0, 0, -30, -30],
            [-50, -30, -30, -30, -30, -30, -30, -50],
        ]

        # Transposition table
        self.transposition_table = {}
        self.max_table_size = 100000

        # Killer moves
        self.killer_moves = defaultdict(list)

        # Position history for repetition detection
        self.position_history = []
        self.position_count = defaultdict(int)
        self.recent_moves = []  # Track last few moves to avoid repetition

        # Search statistics
        self.nodes_searched = 0
        self.cache_hits = 0
        self.cutoffs = 0
        self.quiescence_nodes = 0

        # Time management
        self.search_start_time = 0
        self.time_limit = 0
        self.time_up = False

        # Difficulty settings - IMPROVED
        self.difficulty_config = {
            "medium": {
                "depth": 3,
                "time_limit": 2.0,
                "use_book": True,
                "randomness": 0.3,
            },
            "hard": {
                "depth": 4,
                "time_limit": 4.0,
                "use_book": True,
                "randomness": 0.15,
            },
            "impossible": {
                "depth": 5,
                "time_limit": 7.0,
                "use_book": True,
                "randomness": 0.05,
            },
        }

        # Castling and en passant
        self.castling_rights = {
            "white_kingside": True,
            "white_queenside": True,
            "black_kingside": True,
            "black_queenside": True,
        }
        self.en_passant_target = None

        # Game phase tracking
        self.game_phase = "opening"
        self.move_count = 0

    def reset_game_state(self):
        """Reset game state for new game"""
        self.castling_rights = {
            "white_kingside": True,
            "white_queenside": True,
            "black_kingside": True,
            "black_queenside": True,
        }
        self.en_passant_target = None
        self.killer_moves.clear()
        self.position_history = []
        self.position_count.clear()
        self.recent_moves = []
        self.game_phase = "opening"
        self.move_count = 0
        self.transposition_table.clear()

    def _get_piece_color(self, piece):
        """Get piece color"""
        return "white" if piece.isupper() else "black"

    def is_opposite_color(self, piece1, piece2):
        """Check if two pieces are of opposite color"""
        if not piece1 or not piece2:
            return False
        return self._get_piece_color(piece1) != self._get_piece_color(piece2)

    def is_correct_color(self, piece, color):
        """Check if piece is of the specified color"""
        return self._get_piece_color(piece) == color

    def hash_position(self, board):
        """Create unique hash for board position"""
        board_str = "".join(["".join(row) for row in board])
        rights_str = "".join(
            [
                "K" if self.castling_rights["white_kingside"] else "",
                "Q" if self.castling_rights["white_queenside"] else "",
                "k" if self.castling_rights["black_kingside"] else "",
                "q" if self.castling_rights["black_queenside"] else "",
            ]
        )
        ep_str = str(self.en_passant_target) if self.en_passant_target else ""
        full_str = board_str + rights_str + ep_str
        return hashlib.md5(full_str.encode()).hexdigest()

    def add_position_to_history(self, board):
        """Track position for repetition detection"""
        pos_hash = self.hash_position(board)
        self.position_history.append(pos_hash)
        self.position_count[pos_hash] += 1

    def is_threefold_repetition(self):
        """Check if current position has occurred 3 times"""
        if not self.position_history:
            return False
        current_pos = self.position_history[-1]
        return self.position_count[current_pos] >= 3

    def is_position_repeated(self, board):
        """Check if position has been seen before"""
        pos_hash = self.hash_position(board)
        return self.position_count[pos_hash] >= 1

    def store_position(self, board, depth, score, move_type, best_move=None):
        """Store evaluated position with best move"""
        pos_hash = self.hash_position(board)

        if len(self.transposition_table) > self.max_table_size:
            keys_to_remove = list(self.transposition_table.keys())[:10000]
            for key in keys_to_remove:
                del self.transposition_table[key]

        self.transposition_table[pos_hash] = {
            "depth": depth,
            "score": score,
            "type": move_type,
            "best_move": best_move,
        }

    def lookup_position(self, board, depth):
        """Look up position in transposition table"""
        pos_hash = self.hash_position(board)
        if pos_hash in self.transposition_table:
            entry = self.transposition_table[pos_hash]
            if entry["depth"] >= depth:
                self.cache_hits += 1
                return entry
        return None

    def detect_game_phase(self, board):
        """Detect current game phase"""
        total_material = 0
        piece_count = 0
        queen_count = 0

        for row in board:
            for piece in row:
                if piece and piece.lower() != "k":
                    piece_count += 1
                    total_material += abs(self.piece_values.get(piece, 0))
                    if piece.lower() == "q":
                        queen_count += 1

        if self.move_count < 10 and total_material > 6000:
            return "opening"
        elif total_material < 2500 or (queen_count == 0 and total_material < 3500):
            return "endgame"
        else:
            return "middlegame"

    def find_king(self, board, color):
        """Find the king position"""
        king_piece = "K" if color == "white" else "k"
        for row in range(8):
            for col in range(8):
                if board[row][col] == king_piece:
                    return (row, col)
        return None

    def is_square_attacked(self, board, row, col, by_color):
        """Check if square is attacked"""
        for r in range(8):
            for c in range(8):
                piece = board[r][c]
                if piece and self.is_correct_color(piece, by_color):
                    if self.can_piece_attack(board, r, c, row, col, piece):
                        return True
        return False

    def can_piece_attack(self, board, fromRow, fromCol, toRow, toCol, piece):
        """Check if piece can attack target"""
        if fromRow == toRow and fromCol == toCol:
            return False

        pieceType = piece.lower()
        rowDiff = toRow - fromRow
        colDiff = toCol - fromCol
        absRowDiff = abs(rowDiff)
        absColDiff = abs(colDiff)

        if pieceType == "p":
            direction = -1 if piece.isupper() else 1
            return absColDiff == 1 and rowDiff == direction

        if pieceType == "n":
            return (absRowDiff == 2 and absColDiff == 1) or (
                absRowDiff == 1 and absColDiff == 2
            )

        if pieceType == "b":
            if absRowDiff != absColDiff:
                return False
            return self.is_path_clear(board, fromRow, fromCol, toRow, toCol)

        if pieceType == "r":
            if rowDiff != 0 and colDiff != 0:
                return False
            return self.is_path_clear(board, fromRow, fromCol, toRow, toCol)

        if pieceType == "q":
            if rowDiff != 0 and colDiff != 0 and absRowDiff != absColDiff:
                return False
            return self.is_path_clear(board, fromRow, fromCol, toRow, toCol)

        if pieceType == "k":
            return absRowDiff <= 1 and absColDiff <= 1

        return False

    def is_path_clear(self, board, fromRow, fromCol, toRow, toCol):
        """Check if path is clear"""
        rowStep = 1 if toRow > fromRow else -1 if toRow < fromRow else 0
        colStep = 1 if toCol > fromCol else -1 if toCol < fromCol else 0

        currentRow = fromRow + rowStep
        currentCol = fromCol + colStep

        while currentRow != toRow or currentCol != toCol:
            if board[currentRow][currentCol]:
                return False
            currentRow += rowStep
            currentCol += colStep

        return True

    def is_in_check(self, board, color):
        """Check if king is in check"""
        king_pos = self.find_king(board, color)
        if not king_pos:
            return False
        opponent_color = "black" if color == "white" else "white"
        return self.is_square_attacked(board, king_pos[0], king_pos[1], opponent_color)

    def is_checkmate(self, board, color):
        """Check if color is in checkmate"""
        if not self.is_in_check(board, color):
            return False
        legal_moves = self.get_all_legal_moves(board, color)
        return len(legal_moves) == 0

    def is_stalemate(self, board, color):
        """Check if color is in stalemate"""
        if self.is_in_check(board, color):
            return False
        legal_moves = self.get_all_legal_moves(board, color)
        return len(legal_moves) == 0

    def is_insufficient_material(self, board):
        """Check for insufficient material draw"""
        pieces = []
        for row in board:
            for piece in row:
                if piece and piece.lower() != "k":
                    pieces.append(piece.lower())

        if len(pieces) == 0:
            return True
        if len(pieces) == 1 and pieces[0] in ["b", "n"]:
            return True
        if len(pieces) == 2 and pieces.count("b") == 2:
            return True

        return False

    def would_be_in_check_after_move(
        self, board, from_row, from_col, to_row, to_col, color
    ):
        """Check if move would leave king in check"""
        temp_board = copy.deepcopy(board)
        piece = temp_board[from_row][from_col]

        # Handle en passant capture
        if (
            piece.lower() == "p"
            and to_col != from_col
            and temp_board[to_row][to_col] == ""
        ):
            capture_row = from_row
            temp_board[capture_row][to_col] = ""

        temp_board[to_row][to_col] = piece
        temp_board[from_row][from_col] = ""

        return self.is_in_check(temp_board, color)

    def make_move_simple(self, board, from_r, from_c, to_r, to_c):
        """Apply move and return new board"""
        temp_board = copy.deepcopy(board)
        piece = temp_board[from_r][from_c]

        # En passant capture
        if piece.lower() == "p" and to_c != from_c and temp_board[to_r][to_c] == "":
            capture_row = from_r
            temp_board[capture_row][to_c] = ""

        # Castling
        if piece.lower() == "k" and abs(to_c - from_c) == 2:
            if to_c > from_c:  # Kingside
                rook = temp_board[from_r][7]
                temp_board[from_r][5] = rook
                temp_board[from_r][7] = ""
            else:  # Queenside
                rook = temp_board[from_r][0]
                temp_board[from_r][3] = rook
                temp_board[from_r][0] = ""

        temp_board[to_r][to_c] = piece
        temp_board[from_r][from_c] = ""

        # Pawn promotion
        if piece.lower() == "p":
            if (piece == "P" and to_r == 0) or (piece == "p" and to_r == 7):
                temp_board[to_r][to_c] = "Q" if piece == "P" else "q"

        return temp_board

    def get_all_legal_moves(self, board, color):
        """Get all legal moves for color"""
        all_pseudo_legal_moves = self.get_all_pseudo_legal_moves(board, color)
        legal_moves = []

        for move in all_pseudo_legal_moves:
            from_r = move["from"]["row"]
            from_c = move["from"]["col"]
            to_r = move["to"]["row"]
            to_c = move["to"]["col"]

            if not self.would_be_in_check_after_move(
                board, from_r, from_c, to_r, to_c, color
            ):
                legal_moves.append(move)

        return legal_moves

    def get_all_pseudo_legal_moves(self, board, color):
        """Get all pseudo-legal moves"""
        moves = []
        for r in range(8):
            for c in range(8):
                piece = board[r][c]
                if piece and self.is_correct_color(piece, color):
                    moves.extend(self._get_piece_moves(board, r, c))
        return moves

    def _get_piece_moves(self, board, row, col):
        """Get moves for specific piece"""
        piece = board[row][col]
        piece_lower = piece.lower()

        if piece_lower == "p":
            return self.get_pawn_moves(board, row, col, piece)
        elif piece_lower == "n":
            return self.get_knight_moves(board, row, col, piece)
        elif piece_lower == "b":
            return self.get_bishop_moves(board, row, col, piece)
        elif piece_lower == "r":
            return self.get_rook_moves(board, row, col, piece)
        elif piece_lower == "q":
            return self.get_queen_moves(board, row, col, piece)
        elif piece_lower == "k":
            return self.get_king_moves(board, row, col, piece)
        return []

    def get_pawn_moves(self, board, row, col, piece):
        """Get pawn moves including en passant"""
        moves = []
        direction = -1 if piece.isupper() else 1

        # Forward move
        new_row = row + direction
        if 0 <= new_row < 8 and board[new_row][col] == "":
            moves.append(
                {"from": {"row": row, "col": col}, "to": {"row": new_row, "col": col}}
            )

            # Double move from start
            start_row = 6 if piece.isupper() else 1
            if row == start_row:
                new_row2 = row + (2 * direction)
                if 0 <= new_row2 < 8 and board[new_row2][col] == "":
                    moves.append(
                        {
                            "from": {"row": row, "col": col},
                            "to": {"row": new_row2, "col": col},
                        }
                    )

        # Captures
        for dc in [-1, 1]:
            new_row = row + direction
            new_col = col + dc
            if 0 <= new_row < 8 and 0 <= new_col < 8:
                target = board[new_row][new_col]
                if target and self.is_opposite_color(piece, target):
                    moves.append(
                        {
                            "from": {"row": row, "col": col},
                            "to": {"row": new_row, "col": new_col},
                        }
                    )
                elif self.en_passant_target and self.en_passant_target == (
                    new_row,
                    new_col,
                ):
                    moves.append(
                        {
                            "from": {"row": row, "col": col},
                            "to": {"row": new_row, "col": new_col},
                        }
                    )

        return moves

    def get_knight_moves(self, board, row, col, piece):
        """Get knight moves"""
        moves = []
        knight_offsets = [
            (-2, -1),
            (-2, 1),
            (-1, -2),
            (-1, 2),
            (1, -2),
            (1, 2),
            (2, -1),
            (2, 1),
        ]

        for dr, dc in knight_offsets:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < 8 and 0 <= new_col < 8:
                target = board[new_row][new_col]
                if not target or self.is_opposite_color(piece, target):
                    moves.append(
                        {
                            "from": {"row": row, "col": col},
                            "to": {"row": new_row, "col": new_col},
                        }
                    )
        return moves

    def get_bishop_moves(self, board, row, col, piece):
        """Get bishop moves"""
        return self.get_sliding_moves(
            board, row, col, piece, [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        )

    def get_rook_moves(self, board, row, col, piece):
        """Get rook moves"""
        return self.get_sliding_moves(
            board, row, col, piece, [(-1, 0), (1, 0), (0, -1), (0, 1)]
        )

    def get_queen_moves(self, board, row, col, piece):
        """Get queen moves"""
        return self.get_sliding_moves(
            board,
            row,
            col,
            piece,
            [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)],
        )

    def get_king_moves(self, board, row, col, piece):
        """Get king moves including castling"""
        moves = []
        king_offsets = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]
        color = self._get_piece_color(piece)

        for dr, dc in king_offsets:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < 8 and 0 <= new_col < 8:
                target = board[new_row][new_col]
                if not target or self.is_opposite_color(piece, target):
                    moves.append(
                        {
                            "from": {"row": row, "col": col},
                            "to": {"row": new_row, "col": new_col},
                        }
                    )

        # Castling
        if color == "white" and row == 7 and col == 4:
            if self.castling_rights["white_kingside"]:
                if board[7][5] == "" and board[7][6] == "" and board[7][7] == "R":
                    if not self.is_in_check(board, "white"):
                        if not self.is_square_attacked(board, 7, 5, "black"):
                            if not self.is_square_attacked(board, 7, 6, "black"):
                                moves.append(
                                    {
                                        "from": {"row": 7, "col": 4},
                                        "to": {"row": 7, "col": 6},
                                    }
                                )
            if self.castling_rights["white_queenside"]:
                if (
                    board[7][3] == ""
                    and board[7][2] == ""
                    and board[7][1] == ""
                    and board[7][0] == "R"
                ):
                    if not self.is_in_check(board, "white"):
                        if not self.is_square_attacked(board, 7, 3, "black"):
                            if not self.is_square_attacked(board, 7, 2, "black"):
                                moves.append(
                                    {
                                        "from": {"row": 7, "col": 4},
                                        "to": {"row": 7, "col": 2},
                                    }
                                )

        elif color == "black" and row == 0 and col == 4:
            if self.castling_rights["black_kingside"]:
                if board[0][5] == "" and board[0][6] == "" and board[0][7] == "r":
                    if not self.is_in_check(board, "black"):
                        if not self.is_square_attacked(board, 0, 5, "white"):
                            if not self.is_square_attacked(board, 0, 6, "white"):
                                moves.append(
                                    {
                                        "from": {"row": 0, "col": 4},
                                        "to": {"row": 0, "col": 6},
                                    }
                                )
            if self.castling_rights["black_queenside"]:
                if (
                    board[0][3] == ""
                    and board[0][2] == ""
                    and board[0][1] == ""
                    and board[0][0] == "r"
                ):
                    if not self.is_in_check(board, "black"):
                        if not self.is_square_attacked(board, 0, 3, "white"):
                            if not self.is_square_attacked(board, 0, 2, "white"):
                                moves.append(
                                    {
                                        "from": {"row": 0, "col": 4},
                                        "to": {"row": 0, "col": 2},
                                    }
                                )

        return moves

    def get_sliding_moves(self, board, row, col, piece, directions):
        """Get moves for sliding pieces"""
        moves = []
        for dr, dc in directions:
            for i in range(1, 8):
                new_row, new_col = row + dr * i, col + dc * i
                if not (0 <= new_row < 8 and 0 <= new_col < 8):
                    break
                target = board[new_row][new_col]
                if target:
                    if self.is_opposite_color(piece, target):
                        moves.append(
                            {
                                "from": {"row": row, "col": col},
                                "to": {"row": new_row, "col": new_col},
                            }
                        )
                    break
                else:
                    moves.append(
                        {
                            "from": {"row": row, "col": col},
                            "to": {"row": new_row, "col": new_col},
                        }
                    )
        return moves

    def evaluate_board(self, board):
        """Enhanced board evaluation with learning integration"""
        # Check if we have learned evaluation for this position
        pos_hash = self.hash_position(board)
        remembered = self.strategic_db.recall_position(pos_hash)

        if remembered and remembered["games_seen"] >= 3:
            # Use learned evaluation with decay based on game phase
            learned_score = remembered["eval"]
            static_score = self._static_evaluation(board)
            # Blend: 60% learned, 40% static
            base_score = int(0.6 * learned_score + 0.4 * static_score)
        else:
            base_score = self._static_evaluation(board)

        # Add tactical bonuses
        tactical_score = self._evaluate_tactical_patterns(board)

        # Add strategic bonuses with learned weights
        strategic_score = self._evaluate_strategy(board)

        # Add mobility evaluation
        mobility_score = self._evaluate_mobility(board)

        # Add king safety
        king_safety_score = self._evaluate_king_safety(board)

        total_score = (
            base_score
            + tactical_score
            + strategic_score
            + mobility_score
            + king_safety_score
        )

        # Store this evaluation for learning (occasionally to avoid overhead)
        if self.nodes_searched % 100 == 0:
            self.strategic_db.remember_position(pos_hash, total_score, 1)

        return total_score

    def _static_evaluation(self, board):
        """Static position evaluation with FIXED piece-square tables"""
        score = 0
        piece_count = 0
        white_king_pos = None
        black_king_pos = None
        white_material = 0
        black_material = 0

        for row in range(8):
            for col in range(8):
                piece = board[row][col]
                if not piece:
                    continue

                piece_count += 1
                val = self.piece_values.get(piece, 0)
                score += val

                piece_type = piece.lower()
                if piece_type == "k":
                    if piece.isupper():
                        white_king_pos = (row, col)
                    else:
                        black_king_pos = (row, col)
                else:
                    if piece.isupper():
                        white_material += abs(val)
                    else:
                        black_material += abs(val)

                # Apply piece-square tables CORRECTLY
                is_white = piece.isupper()

                # For white pieces, flip the row (white at bottom = row 7)
                # For black pieces, use row as-is (black at top = row 0)
                pos_row = 7 - row if is_white else row

                if piece_type == "p":
                    bonus = self.pawn_table[pos_row][col]
                elif piece_type == "n":
                    bonus = self.knight_table[pos_row][col]
                elif piece_type == "b":
                    bonus = self.bishop_table[pos_row][col]
                elif piece_type == "r":
                    bonus = self.rook_table[pos_row][col]
                elif piece_type == "q":
                    bonus = self.queen_table[pos_row][col]
                elif piece_type == "k":
                    if piece_count < 10:  # Endgame
                        bonus = self.king_end_table[pos_row][col]
                    else:
                        bonus = self.king_middle_table[pos_row][col]
                else:
                    bonus = 0

                # Apply bonus: positive for black, negative for white
                if is_white:
                    score -= bonus
                else:
                    score += bonus

        # Endgame mop-up evaluation - FIXED
        if black_material > white_material + 200 and black_king_pos and white_king_pos:
            # Push white king to edge
            white_dist_center = max(3 - white_king_pos[0], white_king_pos[0] - 4) + max(
                3 - white_king_pos[1], white_king_pos[1] - 4
            )
            score += white_dist_center * 10

            # Bring kings closer
            dist_between_kings = abs(black_king_pos[0] - white_king_pos[0]) + abs(
                black_king_pos[1] - white_king_pos[1]
            )
            score += (14 - dist_between_kings) * 5

        return score

    def _evaluate_tactical_patterns(self, board):
        """Evaluate tactical patterns with enhanced detection"""
        bonus = 0

        # Detect forks (much higher bonus)
        black_forks = self._detect_forks(board, "black")
        white_forks = self._detect_forks(board, "white")
        bonus += black_forks * 150
        bonus -= white_forks * 150

        # Record successful fork patterns
        if black_forks > 0:
            pos_hash = self.hash_position(board)
            if self.strategic_db.has_similar_tactical_pattern("forks", pos_hash):
                bonus += 50  # Extra bonus for known good pattern

        # Detect pins (higher bonus)
        black_pins = self._detect_pins(board, "black")
        white_pins = self._detect_pins(board, "white")
        bonus += black_pins * 100
        bonus -= white_pins * 100

        # Detect skewers
        black_skewers = self._detect_skewers(board, "black")
        white_skewers = self._detect_skewers(board, "white")
        bonus += black_skewers * 120
        bonus -= white_skewers * 120

        # Detect discovered attacks
        black_discovered = self._detect_discovered_attacks(board, "black")
        white_discovered = self._detect_discovered_attacks(board, "white")
        bonus += black_discovered * 80
        bonus -= white_discovered * 80

        return bonus

    def _detect_forks(self, board, color):
        """Enhanced fork detection"""
        fork_count = 0
        for r in range(8):
            for c in range(8):
                piece = board[r][c]
                if (
                    piece
                    and self.is_correct_color(piece, color)
                    and piece.lower() == "n"
                ):
                    attacked_pieces = []
                    knight_moves = [
                        (-2, -1),
                        (-2, 1),
                        (-1, -2),
                        (-1, 2),
                        (1, -2),
                        (1, 2),
                        (2, -1),
                        (2, 1),
                    ]

                    for dr, dc in knight_moves:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < 8 and 0 <= nc < 8:
                            target = board[nr][nc]
                            if target and self.is_opposite_color(piece, target):
                                # Weight by piece value
                                piece_value = abs(self.piece_values.get(target, 0))
                                attacked_pieces.append((target, piece_value))

                    if len(attacked_pieces) >= 2:
                        # Better fork if attacking high-value pieces
                        total_value = sum(val for _, val in attacked_pieces)
                        if total_value > 600:  # Queen or rook involved
                            fork_count += 2
                        else:
                            fork_count += 1

        return fork_count

    def _detect_pins(self, board, color):
        """Enhanced pin detection"""
        pin_count = 0
        opponent_color = "white" if color == "black" else "black"
        king_pos = self.find_king(board, opponent_color)

        if not king_pos:
            return 0

        for r in range(8):
            for c in range(8):
                piece = board[r][c]
                if piece and self.is_correct_color(piece, color):
                    if piece.lower() in ["b", "r", "q"]:
                        pin_value = self._check_for_pin(board, r, c, king_pos)
                        pin_count += pin_value

        return pin_count

    def _check_for_pin(self, board, r, c, king_pos):
        """Check if piece creates a pin and return value"""
        piece = board[r][c]
        directions = []

        if piece.lower() in ["b", "q"]:
            directions.extend([(-1, -1), (-1, 1), (1, -1), (1, 1)])
        if piece.lower() in ["r", "q"]:
            directions.extend([(-1, 0), (1, 0), (0, -1), (0, 1)])

        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            pinned_piece = None
            pinned_value = 0

            while 0 <= nr < 8 and 0 <= nc < 8:
                target = board[nr][nc]
                if target:
                    if pinned_piece is None:
                        if self.is_opposite_color(piece, target):
                            pinned_piece = (nr, nc)
                            pinned_value = abs(self.piece_values.get(target, 0))
                        else:
                            break
                    else:
                        if (nr, nc) == king_pos:
                            # Pin value based on pinned piece value
                            if pinned_value >= 300:  # Minor piece or better
                                return 2
                            return 1
                        break
                nr += dr
                nc += dc

        return 0

    def _detect_skewers(self, board, color):
        """Detect skewer attacks"""
        skewer_count = 0

        for r in range(8):
            for c in range(8):
                piece = board[r][c]
                if piece and self.is_correct_color(piece, color):
                    if piece.lower() in ["b", "r", "q"]:
                        directions = []
                        if piece.lower() in ["b", "q"]:
                            directions.extend([(-1, -1), (-1, 1), (1, -1), (1, 1)])
                        if piece.lower() in ["r", "q"]:
                            directions.extend([(-1, 0), (1, 0), (0, -1), (0, 1)])

                        for dr, dc in directions:
                            nr, nc = r + dr, c + dc
                            first_piece = None
                            first_value = 0

                            while 0 <= nr < 8 and 0 <= nc < 8:
                                target = board[nr][nc]
                                if target:
                                    if first_piece is None:
                                        if self.is_opposite_color(piece, target):
                                            first_piece = (nr, nc)
                                            first_value = abs(
                                                self.piece_values.get(target, 0)
                                            )
                                        else:
                                            break
                                    else:
                                        if self.is_opposite_color(piece, target):
                                            second_value = abs(
                                                self.piece_values.get(target, 0)
                                            )
                                            # Skewer: high value in front, lower behind
                                            if (
                                                first_value > second_value
                                                and second_value >= 300
                                            ):
                                                skewer_count += 1
                                        break
                                nr += dr
                                nc += dc

        return skewer_count

    def _detect_discovered_attacks(self, board, color):
        """Detect discovered attack opportunities"""
        discovered_count = 0

        # This is a simplified check - full implementation would be more complex
        for r in range(8):
            for c in range(8):
                piece = board[r][c]
                if piece and self.is_correct_color(piece, color):
                    # Check if moving this piece would discover an attack
                    moves = self._get_piece_moves(board, r, c)
                    for move in moves[:3]:  # Check first few moves to save time
                        temp_board = self.make_move_simple(
                            board,
                            move["from"]["row"],
                            move["from"]["col"],
                            move["to"]["row"],
                            move["to"]["col"],
                        )
                        # Check if any of our pieces now attack high-value targets
                        # (simplified check)
                        opponent_color = "white" if color == "black" else "black"
                        king_pos = self.find_king(temp_board, opponent_color)
                        if king_pos and self.is_square_attacked(
                            temp_board, king_pos[0], king_pos[1], color
                        ):
                            discovered_count += 1
                            break

        return discovered_count

    def _evaluate_strategy(self, board):
        """Evaluate strategic elements with learned weights"""
        bonus = 0

        # Get strategic weights from learning
        center_weight = self.strategic_db.get_strategic_weight("center_control")
        piece_activity_weight = self.strategic_db.get_strategic_weight("piece_activity")

        # Center control (weighted)
        center_squares = [(3, 3), (3, 4), (4, 3), (4, 4)]
        extended_center = [
            (2, 2),
            (2, 3),
            (2, 4),
            (2, 5),
            (3, 2),
            (3, 5),
            (4, 2),
            (4, 5),
            (5, 2),
            (5, 3),
            (5, 4),
            (5, 5),
        ]

        for r, c in center_squares:
            piece = board[r][c]
            if piece:
                if self.is_correct_color(piece, "black"):
                    bonus += int(30 * center_weight)
                else:
                    bonus -= int(30 * center_weight)

        for r, c in extended_center:
            piece = board[r][c]
            if piece:
                if self.is_correct_color(piece, "black"):
                    bonus += int(10 * center_weight)
                else:
                    bonus -= int(10 * center_weight)

        # Piece development in opening
        if self.game_phase == "opening":
            back_rank_pieces = 0
            for c in range(8):
                if board[0][c] and self.is_correct_color(board[0][c], "black"):
                    if board[0][c].lower() in ["n", "b"]:
                        back_rank_pieces += 1
            bonus += (8 - back_rank_pieces) * 10

        # Pawn structure evaluation
        pawn_structure_bonus = self._evaluate_pawn_structure(board)
        bonus += pawn_structure_bonus

        # Connected rooks
        rook_bonus = self._evaluate_rook_placement(board)
        bonus += rook_bonus

        return bonus

    def _evaluate_mobility(self, board):
        """Evaluate piece mobility"""
        black_mobility = len(self.get_all_pseudo_legal_moves(board, "black"))
        white_mobility = len(self.get_all_pseudo_legal_moves(board, "white"))

        # More mobility is better (weighted by game phase)
        mobility_weight = 3 if self.game_phase == "middlegame" else 2
        return (black_mobility - white_mobility) * mobility_weight

    def _evaluate_king_safety(self, board):
        """Evaluate king safety"""
        bonus = 0

        black_king_pos = self.find_king(board, "black")
        white_king_pos = self.find_king(board, "white")

        if black_king_pos and self.game_phase != "endgame":
            # Check pawn shield for black
            kr, kc = black_king_pos
            shield_count = 0
            for dc in [-1, 0, 1]:
                if 0 <= kc + dc < 8 and kr + 1 < 8:
                    if board[kr + 1][kc + dc] == "p":
                        shield_count += 1
            bonus += shield_count * 15

        if white_king_pos and self.game_phase != "endgame":
            # Check pawn shield for white
            kr, kc = white_king_pos
            shield_count = 0
            for dc in [-1, 0, 1]:
                if 0 <= kc + dc < 8 and kr - 1 >= 0:
                    if board[kr - 1][kc + dc] == "P":
                        shield_count += 1
            bonus -= shield_count * 15

        return bonus

    def _evaluate_pawn_structure(self, board):
        """Evaluate pawn structure"""
        bonus = 0

        # Doubled pawns penalty
        for col in range(8):
            black_pawns = sum(1 for row in range(8) if board[row][col] == "p")
            white_pawns = sum(1 for row in range(8) if board[row][col] == "P")

            if black_pawns > 1:
                bonus -= (black_pawns - 1) * 15
            if white_pawns > 1:
                bonus += (white_pawns - 1) * 15

        # Passed pawns bonus
        for row in range(8):
            for col in range(8):
                piece = board[row][col]
                if piece and piece.lower() == "p":
                    if self._is_passed_pawn(board, row, col, piece):
                        distance_to_promotion = row if piece == "p" else (7 - row)
                        pawn_bonus = (7 - distance_to_promotion) * 20
                        if piece == "p":
                            bonus += pawn_bonus
                        else:
                            bonus -= pawn_bonus

        return bonus

    def _is_passed_pawn(self, board, row, col, piece):
        """Check if pawn is passed"""
        is_white = piece.isupper()
        direction = -1 if is_white else 1

        # Check if any enemy pawns can block
        check_row = row + direction
        while 0 <= check_row < 8:
            for check_col in [col - 1, col, col + 1]:
                if 0 <= check_col < 8:
                    target = board[check_row][check_col]
                    if (
                        target
                        and target.lower() == "p"
                        and self.is_opposite_color(piece, target)
                    ):
                        return False
            check_row += direction

        return True

    def _evaluate_rook_placement(self, board):
        """Evaluate rook placement"""
        bonus = 0

        # Rooks on open files
        for col in range(8):
            has_pawn = any(
                board[row][col] and board[row][col].lower() == "p" for row in range(8)
            )
            if not has_pawn:
                # Open file
                for row in range(8):
                    if board[row][col] == "r":
                        bonus += 25
                    elif board[row][col] == "R":
                        bonus -= 25

        # Connected rooks (on same rank or file)
        black_rooks = []
        white_rooks = []
        for row in range(8):
            for col in range(8):
                if board[row][col] == "r":
                    black_rooks.append((row, col))
                elif board[row][col] == "R":
                    white_rooks.append((row, col))

        if len(black_rooks) == 2:
            r1, c1 = black_rooks[0]
            r2, c2 = black_rooks[1]
            if r1 == r2 or c1 == c2:
                bonus += 20

        if len(white_rooks) == 2:
            r1, c1 = white_rooks[0]
            r2, c2 = white_rooks[1]
            if r1 == r2 or c1 == c2:
                bonus -= 20

        return bonus

    def minimax(self, board, depth, alpha, beta, maximizing_player):
        """Enhanced minimax with alpha-beta pruning and quiescence search"""
        self.nodes_searched += 1

        # Time check - CRITICAL FIX
        if self.time_up or (time.time() - self.search_start_time) > self.time_limit:
            self.time_up = True
            return 0

        # Check transposition table
        cached = self.lookup_position(board, depth)
        if cached:
            return cached["score"]

        # Terminal conditions
        if self.is_checkmate(board, "white"):
            return 50000  # Black wins
        if self.is_checkmate(board, "black"):
            return -50000  # White wins
        if self.is_stalemate(board, "white") or self.is_stalemate(board, "black"):
            return 0
        if self.is_insufficient_material(board):
            return 0
        if self.is_threefold_repetition():
            return 0  # Avoid repetition

        # Quiescence search at depth 0
        if depth == 0:
            return self.quiescence_search(board, alpha, beta, maximizing_player)

        if maximizing_player:
            max_eval = float("-inf")
            moves = self.get_all_legal_moves(board, "black")

            if not moves:
                if self.is_in_check(board, "black"):
                    return -50000
                return 0

            # Order moves for better pruning
            moves = self.order_moves(board, moves, "black", depth)

            best_move = None
            for move in moves:
                if self.time_up:  # Check time in loop
                    break

                temp_board = self.make_move_simple(
                    board,
                    move["from"]["row"],
                    move["from"]["col"],
                    move["to"]["row"],
                    move["to"]["col"],
                )

                # Avoid repeated positions - ANTI-REPETITION FIX
                if self.is_position_repeated(temp_board):
                    eval_score = max_eval - 200  # Penalty for repetition
                else:
                    eval_score = self.minimax(temp_board, depth - 1, alpha, beta, False)

                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move

                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    self.cutoffs += 1
                    # Store killer move
                    self._store_killer_move(move, depth)
                    break

            self.store_position(board, depth, max_eval, "exact", best_move)
            return max_eval

        else:
            min_eval = float("inf")
            moves = self.get_all_legal_moves(board, "white")

            if not moves:
                if self.is_in_check(board, "white"):
                    return 50000
                return 0

            moves = self.order_moves(board, moves, "white", depth)

            best_move = None
            for move in moves:
                if self.time_up:  # Check time in loop
                    break

                temp_board = self.make_move_simple(
                    board,
                    move["from"]["row"],
                    move["from"]["col"],
                    move["to"]["row"],
                    move["to"]["col"],
                )

                if self.is_position_repeated(temp_board):
                    eval_score = min_eval + 200  # Penalty for repetition
                else:
                    eval_score = self.minimax(temp_board, depth - 1, alpha, beta, True)

                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move

                beta = min(beta, eval_score)
                if beta <= alpha:
                    self.cutoffs += 1
                    self._store_killer_move(move, depth)
                    break

            self.store_position(board, depth, min_eval, "exact", best_move)
            return min_eval

    def quiescence_search(self, board, alpha, beta, maximizing_player, depth=0):
        """Quiescence search to avoid horizon effect - FIXED depth limit"""
        self.quiescence_nodes += 1

        # CRITICAL: Limit quiescence depth properly
        if depth > 3 or self.time_up:
            return self.evaluate_board(board)

        stand_pat = self.evaluate_board(board)

        if maximizing_player:
            if stand_pat >= beta:
                return beta
            if alpha < stand_pat:
                alpha = stand_pat

            # Only consider captures
            captures = self._get_capture_moves(board, "black")
            captures = self._order_captures(board, captures)

            for move in captures[:10]:  # Limit captures checked
                if self.time_up:
                    break

                temp_board = self.make_move_simple(
                    board,
                    move["from"]["row"],
                    move["from"]["col"],
                    move["to"]["row"],
                    move["to"]["col"],
                )

                score = self.quiescence_search(
                    temp_board, alpha, beta, False, depth + 1
                )

                if score >= beta:
                    return beta
                if score > alpha:
                    alpha = score

            return alpha

        else:
            if stand_pat <= alpha:
                return alpha
            if beta > stand_pat:
                beta = stand_pat

            captures = self._get_capture_moves(board, "white")
            captures = self._order_captures(board, captures)

            for move in captures[:10]:  # Limit captures checked
                if self.time_up:
                    break

                temp_board = self.make_move_simple(
                    board,
                    move["from"]["row"],
                    move["from"]["col"],
                    move["to"]["row"],
                    move["to"]["col"],
                )

                score = self.quiescence_search(temp_board, alpha, beta, True, depth + 1)

                if score <= alpha:
                    return alpha
                if score < beta:
                    beta = score

            return beta

    def _get_capture_moves(self, board, color):
        """Get only capture moves for quiescence search"""
        captures = []
        all_moves = self.get_all_legal_moves(board, color)

        for move in all_moves:
            to_row = move["to"]["row"]
            to_col = move["to"]["col"]
            target = board[to_row][to_col]

            # Include captures and pawn promotions
            if target:
                captures.append(move)
            else:
                # Check for en passant
                from_row = move["from"]["row"]
                from_col = move["from"]["col"]
                piece = board[from_row][from_col]
                if piece.lower() == "p" and to_col != from_col:
                    captures.append(move)

        return captures

    def _order_captures(self, board, captures):
        """Order captures by MVV-LVA (Most Valuable Victim - Least Valuable Attacker)"""
        scored_captures = []

        for move in captures:
            from_row = move["from"]["row"]
            from_col = move["from"]["col"]
            to_row = move["to"]["row"]
            to_col = move["to"]["col"]

            attacker = board[from_row][from_col]
            victim = board[to_row][to_col]

            if victim:
                victim_value = abs(self.piece_values.get(victim, 0))
                attacker_value = abs(self.piece_values.get(attacker, 0))
                score = victim_value - (attacker_value // 10)
            else:
                score = 100  # En passant

            scored_captures.append((score, move))

        scored_captures.sort(key=lambda x: x[0], reverse=True)
        return [move for _, move in scored_captures]

    def order_moves(self, board, moves, color, depth):
        """Enhanced move ordering for alpha-beta pruning"""
        if not moves:
            return moves

        scored_moves = []

        # Check transposition table for best move
        cached = self.lookup_position(board, depth)
        best_move_from_cache = (
            cached["best_move"] if cached and "best_move" in cached else None
        )

        for idx, move in enumerate(moves):
            score = 0
            from_r, from_c = move["from"]["row"], move["from"]["col"]
            to_r, to_c = move["to"]["row"], move["to"]["col"]

            # Prioritize transposition table move
            if best_move_from_cache and move == best_move_from_cache:
                score += 10000

            # Captures (MVV-LVA)
            captured = board[to_r][to_c]
            if captured:
                victim_value = abs(self.piece_values.get(captured, 0))
                attacker_value = abs(self.piece_values.get(board[from_r][from_c], 0))
                score += 1000 + victim_value - (attacker_value // 10)

            # Killer moves
            if self._is_killer_move(move, depth):
                score += 500

            # Checks
            temp_board = self.make_move_simple(board, from_r, from_c, to_r, to_c)
            opponent_color = "white" if color == "black" else "black"
            if self.is_in_check(temp_board, opponent_color):
                score += 300

            # Center control
            if to_r in [3, 4] and to_c in [3, 4]:
                score += 50

            # Pawn advancement
            piece = board[from_r][from_c]
            if piece.lower() == "p":
                if piece == "p" and to_r > from_r:
                    score += (to_r - from_r) * 10
                elif piece == "P" and to_r < from_r:
                    score += (from_r - to_r) * 10

            # Castling
            if piece.lower() == "k" and abs(to_c - from_c) == 2:
                score += 100

            scored_moves.append((score, idx, move))

        # Sort by score (descending), then by index for stability
        scored_moves.sort(key=lambda x: (x[0], -x[1]), reverse=True)
        return [move for _, _, move in scored_moves]

    def _store_killer_move(self, move, depth):
        """Store killer move for this depth"""
        if move not in self.killer_moves[depth]:
            self.killer_moves[depth].insert(0, move)
            if len(self.killer_moves[depth]) > 2:
                self.killer_moves[depth].pop()

    def _is_killer_move(self, move, depth):
        """Check if move is a killer move"""
        return move in self.killer_moves.get(depth, [])

    def _move_to_string(self, move):
        """Convert move to string notation"""
        from_r, from_c = move["from"]["row"], move["from"]["col"]
        to_r, to_c = move["to"]["row"], move["to"]["col"]
        from_sq = chr(97 + from_c) + str(8 - from_r)
        to_sq = chr(97 + to_c) + str(8 - to_r)
        return f"{from_sq}{to_sq}"

    def _string_to_move(self, move_str):
        """Convert string notation to move dict"""
        if len(move_str) != 4:
            return None

        from_col = ord(move_str[0]) - 97
        from_row = 8 - int(move_str[1])
        to_col = ord(move_str[2]) - 97
        to_row = 8 - int(move_str[3])

        return {
            "from": {"row": from_row, "col": from_col},
            "to": {"row": to_row, "col": to_col},
        }

    def iterative_deepening(self, board, max_depth, time_limit, color):
        """Iterative deepening search with time management"""
        self.search_start_time = time.time()
        self.time_limit = time_limit
        self.time_up = False

        best_move = None
        best_score = float("-inf") if color == "black" else float("inf")

        for depth in range(1, max_depth + 1):
            if self.time_up:
                break

            legal_moves = self.get_all_legal_moves(board, color)
            if not legal_moves:
                break

            legal_moves = self.order_moves(board, legal_moves, color, depth)

            current_best = None
            current_score = float("-inf") if color == "black" else float("inf")

            for move in legal_moves:
                if self.time_up:
                    break

                temp_board = self.make_move_simple(
                    board,
                    move["from"]["row"],
                    move["from"]["col"],
                    move["to"]["row"],
                    move["to"]["col"],
                )

                if color == "black":
                    score = self.minimax(
                        temp_board, depth - 1, float("-inf"), float("inf"), False
                    )
                    if score > current_score:
                        current_score = score
                        current_best = move
                else:
                    score = self.minimax(
                        temp_board, depth - 1, float("-inf"), float("inf"), True
                    )
                    if score < current_score:
                        current_score = score
                        current_best = move

            if current_best and not self.time_up:
                best_move = current_best
                best_score = current_score

        return best_move, best_score

    def calculate_move(self, board, difficulty="hard"):
        """Main AI move calculation with learning integration - FULLY FIXED"""
        self.nodes_searched = 0
        self.cache_hits = 0
        self.cutoffs = 0
        self.quiescence_nodes = 0
        self.move_count += 1

        # Update game phase
        self.game_phase = self.detect_game_phase(board)

        # Add current position to history
        self.add_position_to_history(board)

        # Get difficulty configuration
        if difficulty in self.difficulty_config:
            config = self.difficulty_config[difficulty]
        else:
            config = self.difficulty_config["hard"]

        max_depth = config["depth"]
        time_limit = config["time_limit"]
        use_opening_book = config["use_book"]
        randomness = config["randomness"]

        legal_moves = self.get_all_legal_moves(board, "black")
        if not legal_moves:
            return None

        # Check opening book first (only in opening phase)
        book_move = None
        if use_opening_book and self.game_phase == "opening" and self.move_count <= 12:
            pos_hash = self.hash_position(board)
            book_move_str = self.strategic_db.get_opening_book_move(pos_hash)

            if book_move_str:
                book_move = self._string_to_move(book_move_str)
                if book_move and any(
                    m["from"] == book_move["from"] and m["to"] == book_move["to"]
                    for m in legal_moves
                ):
                    # Use book move
                    return {
                        "move": book_move,
                        "eval": self.evaluate_board(board),
                        "depth": 0,
                        "nodes": 0,
                        "thought_process": "Using opening book move",
                    }

        # Use iterative deepening for best move
        best_move, best_score = self.iterative_deepening(
            board, max_depth, time_limit, "black"
        )

        if not best_move:
            # Fallback: pick best move from legal moves
            best_move = legal_moves[0]
            best_score = self.evaluate_board(
                self.make_move_simple(
                    board,
                    best_move["from"]["row"],
                    best_move["from"]["col"],
                    best_move["to"]["row"],
                    best_move["to"]["col"],
                )
            )

        # Add some randomness for lower difficulties
        if randomness > 0 and random.random() < randomness:
            # Pick a random good move instead
            candidate_moves = self.order_moves(board, legal_moves, "black", max_depth)[
                :5
            ]
            best_move = random.choice(candidate_moves)

        # Calculate evaluation
        final_eval = self.evaluate_board(
            self.make_move_simple(
                board,
                best_move["from"]["row"],
                best_move["from"]["col"],
                best_move["to"]["row"],
                best_move["to"]["col"],
            )
        )

        thought_process = (
            f"Analyzed {self.nodes_searched} positions "
            f"(depth {max_depth}, {self.cache_hits} cache hits)"
        )

        return {
            "move": best_move,
            "eval": final_eval,
            "depth": max_depth,
            "nodes": self.nodes_searched,
            "thought_process": thought_process,
        }

    def suggest_move_for_player(self, board):
        """Suggest best move for player (white)"""
        legal_moves = self.get_all_legal_moves(board, "white")
        if not legal_moves:
            return None

        # Use shallow search to find best player move
        best_move = None
        best_score = float("inf")

        legal_moves = self.order_moves(board, legal_moves, "white", 3)

        for move in legal_moves[:10]:  # Check top 10 moves
            temp_board = self.make_move_simple(
                board,
                move["from"]["row"],
                move["from"]["col"],
                move["to"]["row"],
                move["to"]["col"],
            )

            score = self.minimax(temp_board, 2, float("-inf"), float("inf"), True)

            if score < best_score:
                best_score = score
                best_move = move

        return best_move

    def learn_from_game(self, move_history, winner, difficulty):
        """Learn from completed game and update strategic database"""

        # Determine outcome from AI perspective
        if winner == "black":
            outcome = "win"
            final_score = 1000
        elif winner == "white":
            outcome = "loss"
            final_score = -1000
        else:
            outcome = "draw"
            final_score = 0

        # Learn from opening moves (first 10 moves)
        opening_moves = move_history[: min(10, len(move_history))]

        # Create a temporary board to replay moves
        temp_board = [
            ["r", "n", "b", "q", "k", "b", "n", "r"],
            ["p", "p", "p", "p", "p", "p", "p", "p"],
            ["", "", "", "", "", "", "", ""],
            ["", "", "", "", "", "", "", ""],
            ["", "", "", "", "", "", "", ""],
            ["", "", "", "", "", "", "", ""],
            ["P", "P", "P", "P", "P", "P", "P", "P"],
            ["R", "N", "B", "Q", "K", "B", "N", "R"],
        ]

        # Reset castling for replay
        saved_castling = self.castling_rights.copy()
        saved_en_passant = self.en_passant_target

        self.castling_rights = {
            "white_kingside": True,
            "white_queenside": True,
            "black_kingside": True,
            "black_queenside": True,
        }
        self.en_passant_target = None

        # Learn from opening
        for idx, move_data in enumerate(opening_moves):
            if not move_data:
                continue

            # Only learn from AI moves (every other move, starting from index 1)
            if idx % 2 == 1:  # AI moves
                pos_hash = self.hash_position(temp_board)
                move_str = self._move_to_string(move_data)

                # Record in opening book
                self.strategic_db.record_opening_move(
                    pos_hash, move_str, outcome, final_score
                )

            # Apply move to temp board
            if "from" in move_data and "to" in move_data:
                temp_board = self.make_move_simple(
                    temp_board,
                    move_data["from"]["row"],
                    move_data["from"]["col"],
                    move_data["to"]["row"],
                    move_data["to"]["col"],
                )

        # Learn from tactical patterns
        if outcome == "win":
            # Record winning sequence (last 6 moves)
            winning_sequence = [
                self._move_to_string(m) for m in move_history[-6:] if m and "from" in m
            ]
            if winning_sequence:
                self.strategic_db.record_winning_sequence(winning_sequence, final_score)

            # Update strategic weights positively
            self.strategic_db.update_strategic_weight("center_control", True)
            self.strategic_db.update_strategic_weight("piece_activity", True)

        elif outcome == "loss":
            # Record losing sequence to avoid
            losing_sequence = [
                self._move_to_string(m) for m in move_history[-6:] if m and "from" in m
            ]
            if losing_sequence:
                self.strategic_db.record_losing_sequence(losing_sequence)

            # Update strategic weights negatively
            self.strategic_db.update_strategic_weight("king_safety", False)

        # Update statistics
        if outcome == "win":
            self.strategic_db.data["stats"]["successful_predictions"] += 1
        elif outcome == "loss":
            self.strategic_db.data["stats"]["failed_predictions"] += 1

        # Save database
        self.strategic_db.save()

        # Restore castling state
        self.castling_rights = saved_castling
        self.en_passant_target = saved_en_passant

        # Return learning stats
        return self.strategic_db.get_learning_stats()

    def load_learning_data(self, data):
        """Load learning data (for compatibility with existing code)"""
        # This is now handled by strategic_db, but keep for compatibility
        if data and "opening_book" in data:
            # Merge with strategic database
            for pos_hash, moves in data.get("opening_book", {}).items():
                if pos_hash not in self.strategic_db.data["opening_book"]:
                    self.strategic_db.data["opening_book"][pos_hash] = moves

    def get_ai_statistics(self):
        """Get comprehensive AI statistics"""
        learning_stats = self.strategic_db.get_learning_stats()

        return {
            "learning": learning_stats,
            "transposition_table_size": len(self.transposition_table),
            "killer_moves_stored": sum(
                len(moves) for moves in self.killer_moves.values()
            ),
            "position_history_length": len(self.position_history),
            "current_game_phase": self.game_phase,
            "moves_played": self.move_count,
        }

    def export_learning_data(self):
        """Export all learning data for backup"""
        return {
            "strategic_database": self.strategic_db.data,
            "game_statistics": self.get_ai_statistics(),
            "export_date": datetime.now().isoformat(),
        }

    def import_learning_data(self, data):
        """Import learning data from backup"""
        if "strategic_database" in data:
            self.strategic_db.data = data["strategic_database"]
            self.strategic_db.save()

    def reset_learning(self):
        """Reset all learning data (use with caution!)"""
        self.strategic_db.reset_database()
        self.transposition_table.clear()
        self.killer_moves.clear()
        print("⚠️ All learning data has been reset!")
