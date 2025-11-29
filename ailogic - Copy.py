import random
import copy
from datetime import datetime


class ChessAI:
    def __init__(self):
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

        # Learning data storage
        self.learning_data = {
            "opening_book": {},
            "position_evaluations": {},
            "winning_patterns": [],
            "losing_patterns": [],
        }

        self.search_depth = {"medium": 3, "hard": 4, "impossible": 5}

        # Track castling rights and en passant
        self.castling_rights = {
            "white_kingside": True,
            "white_queenside": True,
            "black_kingside": True,
            "black_queenside": True,
        }
        self.en_passant_target = None

    def reset_game_state(self):
        """Reset castling and en passant for new game"""
        self.castling_rights = {
            "white_kingside": True,
            "white_queenside": True,
            "black_kingside": True,
            "black_queenside": True,
        }
        self.en_passant_target = None

    def _get_piece_color(self, piece):
        """Get piece color"""
        return "white" if piece.isupper() else "black"

    def is_opposite_color(self, piece1, piece2):
        """Check if two pieces are of opposite color"""
        if not piece1 or not piece2:
            return False
        color1 = self._get_piece_color(piece1)
        color2 = self._get_piece_color(piece2)
        return color1 != color2

    def is_correct_color(self, piece, color):
        """Check if piece is of the specified color"""
        return self._get_piece_color(piece) == color

    def load_learning_data(self, data):
        """Load learned patterns and evaluations"""
        if data:
            self.learning_data = data

    def find_king(self, board, color):
        """Find the king position for a given color"""
        king_piece = "K" if color == "white" else "k"
        for row in range(8):
            for col in range(8):
                if board[row][col] == king_piece:
                    return (row, col)
        return None

    def is_square_attacked(self, board, row, col, by_color):
        """Check if a square is attacked by a given color"""
        for r in range(8):
            for c in range(8):
                piece = board[r][c]
                if piece and self.is_correct_color(piece, by_color):
                    if self.can_piece_attack(board, r, c, row, col, piece):
                        return True
        return False

    def can_piece_attack(self, board, fromRow, fromCol, toRow, toCol, piece):
        """Check if a piece can attack a target square"""
        if fromRow == toRow and fromCol == toCol:
            return False

        pieceType = piece.lower()
        rowDiff = toRow - fromRow
        colDiff = toCol - fromCol
        absRowDiff = abs(rowDiff)
        absColDiff = abs(colDiff)

        # Pawn attacks
        if pieceType == "p":
            direction = -1 if piece.isupper() else 1
            return absColDiff == 1 and rowDiff == direction

        # Knight attacks
        if pieceType == "n":
            return (absRowDiff == 2 and absColDiff == 1) or (
                absRowDiff == 1 and absColDiff == 2
            )

        # Bishop attacks
        if pieceType == "b":
            if absRowDiff != absColDiff:
                return False
            return self.is_path_clear(board, fromRow, fromCol, toRow, toCol)

        # Rook attacks
        if pieceType == "r":
            if rowDiff != 0 and colDiff != 0:
                return False
            return self.is_path_clear(board, fromRow, fromCol, toRow, toCol)

        # Queen attacks
        if pieceType == "q":
            if rowDiff != 0 and colDiff != 0 and absRowDiff != absColDiff:
                return False
            return self.is_path_clear(board, fromRow, fromCol, toRow, toCol)

        # King attacks
        if pieceType == "k":
            return absRowDiff <= 1 and absColDiff <= 1

        return False

    def is_path_clear(self, board, fromRow, fromCol, toRow, toCol):
        """Check if path is clear for sliding pieces"""
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
        """Check if the king of given color is in check"""
        king_pos = self.find_king(board, color)
        if not king_pos:
            return False
        opponent_color = "black" if color == "white" else "white"
        return self.is_square_attacked(board, king_pos[0], king_pos[1], opponent_color)

    def is_checkmate(self, board, color):
        """Check if the given color is in checkmate"""
        if not self.is_in_check(board, color):
            return False
        legal_moves = self.get_all_legal_moves(board, color)
        return len(legal_moves) == 0

    def is_stalemate(self, board, color):
        """Check if the given color is in stalemate"""
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

        # King vs King
        if len(pieces) == 0:
            return True

        # King + Bishop vs King or King + Knight vs King
        if len(pieces) == 1 and pieces[0] in ["b", "n"]:
            return True

        # King + Bishop vs King + Bishop (same color squares)
        if len(pieces) == 2 and pieces.count("b") == 2:
            # Would need to check if bishops are on same color - simplified
            return True

        return False

    def would_be_in_check_after_move(
        self, board, from_row, from_col, to_row, to_col, color
    ):
        """Check if making a move would leave/put the king in check"""
        temp_board = copy.deepcopy(board)
        piece = temp_board[from_row][from_col]

        # Handle en passant capture
        if (
            piece.lower() == "p"
            and to_col != from_col
            and temp_board[to_row][to_col] == ""
        ):
            # En passant capture
            capture_row = from_row
            temp_board[capture_row][to_col] = ""

        temp_board[to_row][to_col] = piece
        temp_board[from_row][from_col] = ""

        return self.is_in_check(temp_board, color)

    def make_move_simple(self, board, from_r, from_c, to_r, to_c):
        """Helper to apply a move on a new board copy"""
        temp_board = copy.deepcopy(board)
        piece = temp_board[from_r][from_c]

        # Handle en passant
        if piece.lower() == "p" and to_c != from_c and temp_board[to_r][to_c] == "":
            # En passant capture
            capture_row = from_r
            temp_board[capture_row][to_c] = ""

        # Handle castling
        if piece.lower() == "k" and abs(to_c - from_c) == 2:
            # Kingside castling
            if to_c > from_c:
                rook = temp_board[from_r][7]
                temp_board[from_r][5] = rook
                temp_board[from_r][7] = ""
            # Queenside castling
            else:
                rook = temp_board[from_r][0]
                temp_board[from_r][3] = rook
                temp_board[from_r][0] = ""

        temp_board[to_r][to_c] = piece
        temp_board[from_r][from_c] = ""

        # Handle pawn promotion
        if piece.lower() == "p":
            if (piece == "P" and to_r == 0) or (piece == "p" and to_r == 7):
                # Auto-promote to queen
                temp_board[to_r][to_c] = "Q" if piece == "P" else "q"

        return temp_board

    def get_all_legal_moves(self, board, color):
        """Get all LEGAL moves for the given color"""
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
        """Collects all pseudo-legal moves for 'color'"""
        moves = []
        for r in range(8):
            for c in range(8):
                piece = board[r][c]
                if piece and self.is_correct_color(piece, color):
                    moves.extend(self._get_piece_moves(board, r, c))
        return moves

    def _get_piece_moves(self, board, row, col):
        """Helper to route to correct piece move generator"""
        piece = board[row][col]
        piece_lower = piece.lower()

        if piece_lower == "p":
            moves = self.get_pawn_moves(board, row, col, piece)
        elif piece_lower == "n":
            moves = self.get_knight_moves(board, row, col, piece)
        elif piece_lower == "b":
            moves = self.get_bishop_moves(board, row, col, piece)
        elif piece_lower == "r":
            moves = self.get_rook_moves(board, row, col, piece)
        elif piece_lower == "q":
            moves = self.get_queen_moves(board, row, col, piece)
        elif piece_lower == "k":
            moves = self.get_king_moves(board, row, col, piece)
        else:
            moves = []

        return moves

    def get_pawn_moves(self, board, row, col, piece):
        """Get pawn moves including en passant"""
        moves = []
        direction = -1 if piece.isupper() else 1
        color = self._get_piece_color(piece)

        # Forward move
        new_row = row + direction
        if 0 <= new_row < 8 and board[new_row][col] == "":
            moves.append(
                {"from": {"row": row, "col": col}, "to": {"row": new_row, "col": col}}
            )

            # Double move from starting position
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

        # Diagonal captures
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
                # En passant
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
        king_moves = [
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

        for dr, dc in king_moves:
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
            # Kingside castling
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
            # Queenside castling
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
            # Kingside castling
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
            # Queenside castling
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
        """Evaluate board from black's perspective"""
        score = 0
        piece_count = 0

        for row in range(8):
            for col in range(8):
                piece = board[row][col]
                if not piece:
                    continue

                piece_count += 1
                score += self.piece_values.get(piece, 0)

                piece_lower = piece.lower()
                is_white = piece.isupper()
                pos_row = 7 - row if is_white else row

                if piece_lower == "p":
                    bonus = self.pawn_table[pos_row][col]
                elif piece_lower == "n":
                    bonus = self.knight_table[pos_row][col]
                elif piece_lower == "b":
                    bonus = self.bishop_table[pos_row][col]
                elif piece_lower == "r":
                    bonus = self.rook_table[pos_row][col]
                elif piece_lower == "q":
                    bonus = self.queen_table[pos_row][col]
                elif piece_lower == "k":
                    # Use endgame table if few pieces
                    if piece_count < 14:
                        bonus = self.king_end_table[pos_row][col]
                    else:
                        bonus = self.king_middle_table[pos_row][col]
                else:
                    bonus = 0

                if is_white:
                    score -= bonus
                else:
                    score += bonus

        return score

    def order_moves(self, board, moves, color):
        """Order moves for better alpha-beta pruning"""

        def move_priority(move):
            priority = 0
            from_r, from_c = move["from"]["row"], move["from"]["col"]
            to_r, to_c = move["to"]["row"], move["to"]["col"]

            piece = board[from_r][from_c]
            target = board[to_r][to_c]

            # Prioritize captures
            if target:
                victim_value = abs(self.piece_values.get(target, 0))
                attacker_value = abs(self.piece_values.get(piece, 0))
                priority += (victim_value - attacker_value / 10) * 10

            # Prioritize center control
            center_distance = abs(3.5 - to_r) + abs(3.5 - to_c)
            priority -= center_distance

            # Prioritize pawn promotion
            if piece.lower() == "p":
                if (piece == "P" and to_r == 0) or (piece == "p" and to_r == 7):
                    priority += 900

            return priority

        return sorted(moves, key=move_priority, reverse=True)

    def minimax(self, board, depth, alpha, beta, is_maximizing):
        """Minimax with alpha-beta pruning"""
        if depth == 0:
            return self.evaluate_board(board)

        color = "black" if is_maximizing else "white"

        if self.is_checkmate(board, color):
            return -10000 if is_maximizing else 10000

        if self.is_stalemate(board, color) or self.is_insufficient_material(board):
            return 0

        legal_moves = self.get_all_legal_moves(board, color)
        if not legal_moves:
            return 0

        # Order moves for better pruning
        legal_moves = self.order_moves(board, legal_moves, color)

        if is_maximizing:
            max_eval = float("-inf")
            for move in legal_moves:
                new_board = self.make_move_simple(
                    board,
                    move["from"]["row"],
                    move["from"]["col"],
                    move["to"]["row"],
                    move["to"]["col"],
                )
                eval_score = self.minimax(new_board, depth - 1, alpha, beta, False)
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float("inf")
            for move in legal_moves:
                new_board = self.make_move_simple(
                    board,
                    move["from"]["row"],
                    move["from"]["col"],
                    move["to"]["row"],
                    move["to"]["col"],
                )
                eval_score = self.minimax(new_board, depth - 1, alpha, beta, True)
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval

    def calculate_move(self, board, difficulty):
        """Main AI move calculation"""
        depth = self.search_depth.get(difficulty, 4)
        legal_moves = self.get_all_legal_moves(board, "black")

        if not legal_moves:
            return None

        # Order moves
        legal_moves = self.order_moves(board, legal_moves, "black")

        best_move = None
        best_value = float("-inf")
        alpha = float("-inf")
        beta = float("inf")

        for move in legal_moves:
            new_board = self.make_move_simple(
                board,
                move["from"]["row"],
                move["from"]["col"],
                move["to"]["row"],
                move["to"]["col"],
            )
            move_value = self.minimax(new_board, depth - 1, alpha, beta, False)

            if move_value > best_value:
                best_value = move_value
                best_move = move

            alpha = max(alpha, move_value)

        piece = board[best_move["from"]["row"]][best_move["from"]["col"]]
        thought = (
            f"Evaluated {len(legal_moves)} moves | Depth: {depth} | Best: {best_value}"
        )

        return {"move": best_move, "evaluation": best_value, "thought_process": thought}

    def suggest_move_for_player(self, board):
        """Suggest move for player"""
        depth = 3
        legal_moves = self.get_all_legal_moves(board, "white")

        if not legal_moves:
            return None

        legal_moves = self.order_moves(board, legal_moves, "white")
        best_move = None
        best_value = float("inf")

        for move in legal_moves:
            new_board = self.make_move_simple(
                board,
                move["from"]["row"],
                move["from"]["col"],
                move["to"]["row"],
                move["to"]["col"],
            )
            move_value = self.minimax(
                new_board, depth - 1, float("-inf"), float("inf"), True
            )

            if move_value < best_value:
                best_value = move_value
                best_move = move

        return best_move

    def learn_from_game(self, moves, winner, difficulty):
        """Learn from completed game"""
        if winner == "black":
            self.learning_data["winning_patterns"].append(
                {
                    "moves": moves[-10:],
                    "difficulty": difficulty,
                    "date": datetime.now().isoformat(),
                }
            )
        elif winner == "white":
            self.learning_data["losing_patterns"].append(
                {
                    "moves": moves[-10:],
                    "difficulty": difficulty,
                    "date": datetime.now().isoformat(),
                }
            )

        if len(self.learning_data["winning_patterns"]) > 100:
            self.learning_data["winning_patterns"] = self.learning_data[
                "winning_patterns"
            ][-100:]
        if len(self.learning_data["losing_patterns"]) > 100:
            self.learning_data["losing_patterns"] = self.learning_data[
                "losing_patterns"
            ][-100:]

        return self.learning_data
