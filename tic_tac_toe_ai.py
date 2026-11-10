"""
Tic-Tac-Toe AI
==============
An unbeatable Tic-Tac-Toe agent built with the Minimax algorithm and
Alpha-Beta Pruning.

Board layout (positions 1-9, entered by the human player):

     1 | 2 | 3
    -----------
     4 | 5 | 6
    -----------
     7 | 8 | 9

Run this file directly to play against the AI in the terminal:
    python tic_tac_toe_ai.py
"""

import math
import random

HUMAN = "X"
AI = "O"
EMPTY = " "

WIN_LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),   # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),   # columns
    (0, 4, 8), (2, 4, 6),              # diagonals
]


# --------------------------------------------------------------------------
# Board helpers
# --------------------------------------------------------------------------

def new_board():
    return [EMPTY] * 9


def print_board(board):
    def cell(i):
        return board[i] if board[i] != EMPTY else str(i + 1)

    rows = [
        f" {cell(0)} | {cell(1)} | {cell(2)} ",
        f" {cell(3)} | {cell(4)} | {cell(5)} ",
        f" {cell(6)} | {cell(7)} | {cell(8)} ",
    ]
    print("\n" + f"\n{'-' * 11}\n".join(rows) + "\n")


def available_moves(board):
    return [i for i, v in enumerate(board) if v == EMPTY]


def winner(board):
    """Return 'X', 'O', 'draw', or None (game still in progress)."""
    for a, b, c in WIN_LINES:
        if board[a] != EMPTY and board[a] == board[b] == board[c]:
            return board[a]
    if EMPTY not in board:
        return "draw"
    return None


def game_over(board):
    return winner(board) is not None


# --------------------------------------------------------------------------
# Minimax with Alpha-Beta Pruning
# --------------------------------------------------------------------------
#
# The AI (maximizing player) tries to maximize its score; the human
# (minimizing player) is assumed to play optimally against it, so the AI
# always plans for the worst case. Scores favor faster wins and slower
# losses by factoring in search depth:
#
#   AI wins    -> +10 - depth
#   Human wins -> depth - 10
#   Draw       ->  0
#
# Alpha-beta pruning cuts off branches that cannot possibly change the
# final decision, which lets the search reach the end of the game tree
# quickly without changing the result.

nodes_evaluated = 0  # simple counter, useful for demonstrating pruning


def minimax(board, depth, is_maximizing, alpha, beta):
    global nodes_evaluated
    nodes_evaluated += 1

    result = winner(board)
    if result == AI:
        return 10 - depth
    if result == HUMAN:
        return depth - 10
    if result == "draw":
        return 0

    if is_maximizing:
        best_score = -math.inf
        for move in available_moves(board):
            board[move] = AI
            score = minimax(board, depth + 1, False, alpha, beta)
            board[move] = EMPTY
            best_score = max(best_score, score)
            alpha = max(alpha, best_score)
            if beta <= alpha:
                break  # beta cutoff
        return best_score
    else:
        best_score = math.inf
        for move in available_moves(board):
            board[move] = HUMAN
            score = minimax(board, depth + 1, True, alpha, beta)
            board[move] = EMPTY
            best_score = min(best_score, score)
            beta = min(beta, best_score)
            if beta <= alpha:
                break  # alpha cutoff
        return best_score


def best_ai_move(board):
    """Return the optimal move index for the AI using minimax search."""
    global nodes_evaluated
    nodes_evaluated = 0

    best_score = -math.inf
    best_move = None
    for move in available_moves(board):
        board[move] = AI
        score = minimax(board, 0, False, -math.inf, math.inf)
        board[move] = EMPTY
        if score > best_score:
            best_score = score
            best_move = move

    return best_move, best_score, nodes_evaluated


# --------------------------------------------------------------------------
# Difficulty levels (optional twist: makes the AI beatable on purpose)
# --------------------------------------------------------------------------

def ai_move(board, difficulty="unbeatable"):
    """
    difficulty: "unbeatable", "hard" (90% optimal), "medium" (60% optimal),
                or "easy" (always random)
    """
    if difficulty == "easy":
        return random.choice(available_moves(board))

    move, score, nodes = best_ai_move(board)

    chance = {"unbeatable": 1.0, "hard": 0.9, "medium": 0.6}.get(difficulty, 1.0)
    if random.random() > chance:
        move = random.choice(available_moves(board))

    return move


# --------------------------------------------------------------------------
# Game loop
# --------------------------------------------------------------------------

def choose_difficulty():
    print("Choose difficulty:")
    print("  1. Unbeatable")
    print("  2. Hard")
    print("  3. Medium")
    print("  4. Easy")
    choice = input("> ").strip()
    return {"1": "unbeatable", "2": "hard", "3": "medium", "4": "easy"}.get(choice, "unbeatable")


def get_human_move(board):
    while True:
        raw = input("Your move (1-9): ").strip()
        if not raw.isdigit() or not (1 <= int(raw) <= 9):
            print("Please enter a number between 1 and 9.")
            continue
        pos = int(raw) - 1
        if board[pos] != EMPTY:
            print("That cell is already taken.")
            continue
        return pos


def play_game():
    print("=" * 40)
    print("  TIC-TAC-TOE — You are X, AI is O")
    print("=" * 40)
    difficulty = choose_difficulty()

    board = new_board()
    print_board(board)

    turn = HUMAN
    while not game_over(board):
        if turn == HUMAN:
            move = get_human_move(board)
            board[move] = HUMAN
        else:
            print("AI is thinking...")
            move, score, nodes = best_ai_move(board) if difficulty == "unbeatable" else (ai_move(board, difficulty), None, None)
            board[move] = AI
            if score is not None:
                print(f"AI plays position {move + 1} (score {score}, {nodes} positions evaluated)")
            else:
                print(f"AI plays position {move + 1}")

        print_board(board)
        turn = AI if turn == HUMAN else HUMAN

    result = winner(board)
    if result == "draw":
        print("It's a draw!")
    elif result == HUMAN:
        print("You win! (shouldn't happen on Unbeatable mode...)")
    else:
        print("AI wins!")


if __name__ == "__main__":
    while True:
        play_game()
        again = input("\nPlay again? (y/n): ").strip().lower()
        if again != "y":
            print("Thanks for playing!")
            break
