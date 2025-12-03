from flask import Flask, render_template, request, jsonify
import json
import os
from datetime import datetime
from ailogic import ChessAI

app = Flask(__name__)

# Initialize AI
chess_ai = ChessAI()

# Data file path
DATA_FILE = "data/chess_ai_brain.json"

# Ensure data directory exists
os.makedirs("data", exist_ok=True)

# Initialize data file if it doesn't exist
if not os.path.exists(DATA_FILE):
    initial_data = {
        "games_played": 0,
        "ai_wins": 0,
        "player_wins": 0,
        "draws": 0,
        "game_history": [],
        "ai_weights": {},
        "learning_data": {
            "opening_book": {},
            "position_evaluations": {},
            "winning_patterns": [],
            "losing_patterns": [],
        },
    }
    with open(DATA_FILE, "w") as f:
        json.dump(initial_data, f, indent=4)


def load_data():
    """Load existing data"""
    try:
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    except:
        return {
            "games_played": 0,
            "ai_wins": 0,
            "player_wins": 0,
            "draws": 0,
            "game_history": [],
            "ai_weights": {},
            "learning_data": {
                "opening_book": {},
                "position_evaluations": {},
                "winning_patterns": [],
                "losing_patterns": [],
            },
        }


def save_data(data):
    """Save data"""
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=4)


# Current game state
current_game = {
    "board": None,
    "moves": [],
    "difficulty": "hard",
    "start_time": None,
    "move_history": [],
}


def make_move_on_board(board, from_r, from_c, to_r, to_c):
    """
    Apply a move on the board and return captured piece, en passant info, and castling updates.
    This handles pawn promotion, en passant, and castling in the actual game.
    """
    piece = board[from_r][from_c]
    captured_piece = board[to_r][to_c]

    # Handle en passant capture
    en_passant_capture = None
    if piece.lower() == "p" and to_c != from_c and captured_piece == "":
        # This is en passant
        capture_row = from_r
        en_passant_capture = board[capture_row][to_c]
        board[capture_row][to_c] = ""
        captured_piece = en_passant_capture

    # Update en passant target for next move
    chess_ai.en_passant_target = None
    if piece.lower() == "p" and abs(to_r - from_r) == 2:
        # Pawn moved 2 squares, set en passant target
        direction = -1 if piece.isupper() else 1
        chess_ai.en_passant_target = (from_r + direction, from_c)

    # Handle castling
    if piece.lower() == "k" and abs(to_c - from_c) == 2:
        # Kingside castling
        if to_c > from_c:
            rook = board[from_r][7]
            board[from_r][5] = rook
            board[from_r][7] = ""
        # Queenside castling
        else:
            rook = board[from_r][0]
            board[from_r][3] = rook
            board[from_r][0] = ""

    # Move the piece
    board[to_r][to_c] = piece
    board[from_r][from_c] = ""

    # Handle pawn promotion (auto-promote to Queen)
    promotion = None
    if piece.lower() == "p":
        if (piece == "P" and to_r == 0) or (piece == "p" and to_r == 7):
            promoted_piece = "Q" if piece == "P" else "q"
            board[to_r][to_c] = promoted_piece
            promotion = promoted_piece

    # Update castling rights
    if piece == "K":
        chess_ai.castling_rights["white_kingside"] = False
        chess_ai.castling_rights["white_queenside"] = False
    elif piece == "k":
        chess_ai.castling_rights["black_kingside"] = False
        chess_ai.castling_rights["black_queenside"] = False
    elif piece == "R":
        if from_r == 7 and from_c == 7:
            chess_ai.castling_rights["white_kingside"] = False
        elif from_r == 7 and from_c == 0:
            chess_ai.castling_rights["white_queenside"] = False
    elif piece == "r":
        if from_r == 0 and from_c == 7:
            chess_ai.castling_rights["black_kingside"] = False
        elif from_r == 0 and from_c == 0:
            chess_ai.castling_rights["black_queenside"] = False

    return {
        "captured": captured_piece,
        "promotion": promotion,
        "en_passant_capture": en_passant_capture,
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/stats", methods=["GET"])
def get_stats():
    """Get AI statistics"""
    data = load_data()

    total_games = data["games_played"]
    ai_wins = data["ai_wins"]
    player_wins = data["player_wins"]
    draws = data["draws"]

    if total_games > 0:
        win_rate = round((ai_wins / total_games) * 100, 1)
    else:
        win_rate = 0

    return jsonify(
        {
            "games": total_games,
            "ai_wins": ai_wins,
            "player_wins": player_wins,
            "draws": draws,
            "win_rate": win_rate,
        }
    )


@app.route("/new_game", methods=["POST"])
def new_game():
    """Start a new game"""
    global current_game

    # Reset castling and en passant
    chess_ai.reset_game_state()

    # Reset current game
    current_game = {
        "board": [
            ["r", "n", "b", "q", "k", "b", "n", "r"],
            ["p", "p", "p", "p", "p", "p", "p", "p"],
            ["", "", "", "", "", "", "", ""],
            ["", "", "", "", "", "", "", ""],
            ["", "", "", "", "", "", "", ""],
            ["", "", "", "", "", "", "", ""],
            ["P", "P", "P", "P", "P", "P", "P", "P"],
            ["R", "N", "B", "Q", "K", "B", "N", "R"],
        ],
        "moves": [],
        "difficulty": "hard",
        "start_time": datetime.now().isoformat(),
        "move_history": [],
    }

    # Load AI learning data
    data = load_data()
    if "learning_data" in data:
        chess_ai.load_learning_data(data["learning_data"])

    # Get stats
    stats = get_stats().json

    return jsonify({"success": True, "stats": stats})


@app.route("/validate_move", methods=["POST"])
def validate_move():
    """Validate if a move is legal"""
    try:
        request_data = request.get_json()
        board = request_data.get("board")
        from_pos = request_data.get("from")
        to_pos = request_data.get("to")

        from_row = from_pos["row"]
        from_col = from_pos["col"]
        to_row = to_pos["row"]
        to_col = to_pos["col"]

        piece = board[from_row][from_col]
        if not piece:
            return jsonify({"valid": False, "reason": "No piece at source"})

        player_color = "white"

        # Get all legal moves for the player
        all_legal_moves = chess_ai.get_all_legal_moves(board, player_color)

        # Filter moves for the specific piece
        piece_legal_moves = [
            move
            for move in all_legal_moves
            if move["from"]["row"] == from_row and move["from"]["col"] == from_col
        ]

        # Check if the requested move is valid
        is_valid = False
        for move in piece_legal_moves:
            if move["to"]["row"] == to_row and move["to"]["col"] == to_col:
                is_valid = True
                break

        if not is_valid and (from_row != to_row or from_col != to_col):
            if chess_ai.is_in_check(board, player_color):
                reason = "Your king is in CHECK! You must escape check."
            elif chess_ai.would_be_in_check_after_move(
                board, from_row, from_col, to_row, to_col, player_color
            ):
                reason = "This move would put your king in check."
            else:
                reason = "Invalid move for this piece."

            return jsonify(
                {"valid": False, "reason": reason, "moves_for_piece": piece_legal_moves}
            )

        # Check if move puts opponent in check
        opponent_in_check = False
        if is_valid:
            temp_board = [row[:] for row in board]
            make_move_on_board(temp_board, from_row, from_col, to_row, to_col)
            opponent_in_check = chess_ai.is_in_check(temp_board, "black")

        return jsonify(
            {
                "valid": is_valid,
                "moves_for_piece": piece_legal_moves,
                "opponent_in_check": opponent_in_check,
            }
        )

    except Exception as e:
        print(f"Error in validate_move: {e}")
        import traceback

        traceback.print_exc()
        return jsonify({"valid": False, "reason": str(e), "moves_for_piece": []})


@app.route("/ai_move", methods=["POST"])
def ai_move():
    """AI makes a move"""
    global current_game
    try:
        request_data = request.get_json()
        board = request_data.get("board")
        difficulty = request_data.get("difficulty") or current_game.get(
            "difficulty", "hard"
        )

        # Check for game end after player's move
        if chess_ai.is_checkmate(board, "black"):
            winner = "white"
            reason = "checkmate"
            save_game_and_learn(winner)
            return jsonify(
                {
                    "success": True,
                    "game_over": True,
                    "winner": winner,
                    "reason": reason,
                    "board": board,
                    "stats": get_stats().json,
                    "player_in_check": False,
                    "ai_in_check": False,
                }
            )
        elif chess_ai.is_stalemate(board, "black"):
            winner = "draw"
            reason = "stalemate"
            save_game_and_learn(winner)
            return jsonify(
                {
                    "success": True,
                    "game_over": True,
                    "winner": winner,
                    "reason": reason,
                    "board": board,
                    "stats": get_stats().json,
                    "player_in_check": False,
                    "ai_in_check": False,
                }
            )

        # AI calculates and makes a move
        ai_move_result = chess_ai.calculate_move(board, difficulty)

        if ai_move_result and ai_move_result.get("move"):
            move = ai_move_result["move"]

            from_row = move["from"]["row"]
            from_col = move["from"]["col"]
            to_row = move["to"]["row"]
            to_col = move["to"]["col"]

            # Apply move with special move handling
            move_result = make_move_on_board(board, from_row, from_col, to_row, to_col)
            captured_piece = move_result["captured"]

            # Record move
            current_game["move_history"].append(
                {
                    "from": {"row": from_row, "col": from_col},
                    "to": {"row": to_row, "col": to_col},
                    "piece": board[to_row][to_col],
                    "captured": captured_piece,
                    "promotion": move_result["promotion"],
                }
            )

            # Check for game end after AI's move
            player_in_check = chess_ai.is_in_check(board, "white")

            game_over = False
            winner = None
            reason = None

            if chess_ai.is_checkmate(board, "white"):
                game_over = True
                winner = "black"
                reason = "checkmate"
                save_game_and_learn(winner)
            elif chess_ai.is_stalemate(board, "white"):
                game_over = True
                winner = "draw"
                reason = "stalemate"
                save_game_and_learn(winner)
            elif chess_ai.is_insufficient_material(board):
                game_over = True
                winner = "draw"
                reason = "insufficient_material"
                save_game_and_learn(winner)

            stats = get_stats().json

            return jsonify(
                {
                    "success": True,
                    "move": move,
                    "board": board,
                    "game_over": game_over,
                    "winner": winner,
                    "reason": reason,
                    "player_in_check": player_in_check,
                    "ai_in_check": chess_ai.is_in_check(board, "black"),
                    "stats": stats,
                    "ai_thought_process": ai_move_result.get("thought_process", ""),
                    "captured_piece": captured_piece if captured_piece else None,
                    "promotion": move_result["promotion"],
                }
            )

        else:
            # No valid moves for AI
            if chess_ai.is_in_check(board, "black"):
                winner = "white"
                reason = "checkmate"
            else:
                winner = "draw"
                reason = "stalemate"

            save_game_and_learn(winner)
            return jsonify(
                {
                    "success": True,
                    "game_over": True,
                    "winner": winner,
                    "reason": reason,
                    "board": board,
                    "stats": get_stats().json,
                    "player_in_check": chess_ai.is_in_check(board, "white"),
                    "ai_thought_process": "No legal moves found. Game Over.",
                }
            )

    except Exception as e:
        print(f"Error in ai_move: {e}")
        import traceback

        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


def save_game_and_learn(winner):
    """Save game data and let AI learn from it"""
    global current_game

    data = load_data()
    data["games_played"] += 1

    if winner == "black":
        data["ai_wins"] += 1
    elif winner == "white":
        data["player_wins"] += 1
    else:
        data["draws"] += 1

    game_record = {
        "game_id": data["games_played"],
        "date": datetime.now().isoformat(),
        "difficulty": current_game["difficulty"],
        "moves": current_game["move_history"],
        "winner": winner,
        "total_moves": len(current_game["move_history"]),
    }

    data["game_history"].append(game_record)

    # Let AI learn
    learning_stats = chess_ai.learn_from_game(
        current_game["move_history"], winner, current_game["difficulty"]
    )
    # Store learning stats (AI manages its own strategic_brain.json)
    if learning_stats:
        data["last_learning_stats"] = learning_stats

    # Keep only last 1000 games
    if len(data["game_history"]) > 1000:
        data["game_history"] = data["game_history"][-1000:]

    save_data(data)
    print(
        f"Game saved! Total: {data['games_played']}, AI: {data['ai_wins']}, Player: {data['player_wins']}"
    )


@app.route("/get_hint", methods=["POST"])
def get_hint():
    """Get a hint for the player"""
    try:
        request_data = request.get_json()
        board = request_data.get("board")

        hint_move = chess_ai.suggest_move_for_player(board)
        return jsonify({"success": True, "hint": hint_move})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/export_data", methods=["GET"])
def export_data():
    """Export all learning data"""
    data = load_data()
    return jsonify(data)


if __name__ == "__main__":
    print("=" * 50)
    print("🎮 Adaptive Chess AI Server Starting...")
    print("=" * 50)
    print("📊 Loading AI brain data...")

    stats = load_data()
    print(f"✅ Games Played: {stats['games_played']}")
    print(f"🤖 AI Wins: {stats['ai_wins']}")
    print(f"👤 Player Wins: {stats['player_wins']}")
    print(f"🤝 Draws: {stats['draws']}")
    print("=" * 50)
    print("🚀 Server running on http://127.0.0.1:5003")
    print("🎯 Open your browser and start playing!")
    print("=" * 50)

    app.run(debug=True, host="0.0.0.0", port=5003)
