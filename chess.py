# Author: Arash Tashakori - arashtash.github.io

import pygame
import chess
import multiprocessing
from collections import defaultdict
import time
import random
import math

pygame.init()

WIDTH, HEIGHT = 640, 640
N = 8                       #N*N board
SQUARE_SIZE = WIDTH // N
COLOURS = [(240, 217, 181), (181, 136, 99)]
IMAGES_OF_PIECES = {}
FPS = 30

#This method loads the piece images
def load_images():
    pieces = ['br', 'bn', 'bb', 'bq', 'bk', 'bp',
              'wr', 'wn', 'wb', 'wq', 'wk', 'wp']

    for piece in pieces:
        IMAGES_OF_PIECES[piece] = pygame.transform.smoothscale(
            pygame.image.load(f"assets/{piece}.png"), (SQUARE_SIZE, SQUARE_SIZE)
        )

#This method draws the chessboard on the screen using the given colours
def draw_board(screen):
    for row in range(N):
        for col in range(N):
            colour = COLOURS[(row + col) % 2]
            pygame.draw.rect(
                screen,
                colour,
                pygame.Rect(col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
            )

#This method uses the current board state to draw pieces on their respective squares on the board
def draw_pieces(screen, board):
    for row in range(N):
        for col in range(N):
            piece = board.piece_at(chess.square(col, 7 - row))
            if piece:
                color = 'w' if piece.color else 'b' #Get the piece colour
                piece_type = piece.symbol().lower() #Get the piece type
                piece_key = f"{color}{piece_type}"  #Combine them to determine the piece

                screen.blit(
                    IMAGES_OF_PIECES[piece_key],
                    pygame.Rect(col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
                )



###############

#Piece values for evaluation
PIECE_VALUES = {
    'p': 100, 'n': 320, 'b': 330, 'r': 500, 'q': 1100, 'k': 20000
}

#These values encourage positional play for each piece type.
PIECE_SQUARE_TABLES = {
    'p': [
         0,   5,  10,   0,   5,  10,   5,   0,
        50,  50,  50,  50,  50,  50,  50,  50,
        10,  10,  20,  30,  30,  20,  10,  10,
         5,   5,  10,  25,  25,  10,   5,   5,
         0,   0,   0,  20,  20,   0,   0,   0,
         5,  -5, -10,   0,   0, -10,  -5,   5,
         5,  10,  10, -20, -20,  10,  10,   5,
         0,   0,   0,   0,   0,   0,   0,   0
    ],
    'n': [
       -50, -40, -30, -30, -30, -30, -40, -50,
       -40, -20,   0,   0,   0,   0, -20, -40,
       -30,   0,  10,  15,  15,  10,   0, -30,
       -30,   5,  15,  20,  20,  15,   5, -30,
       -30,   0,  15,  20,  20,  15,   0, -30,
       -30,   5,  10,  15,  15,  10,   5, -30,
       -40, -20,   0,   5,   5,   0, -20, -40,
       -50, -40, -30, -30, -30, -30, -40, -50
    ],
    'b': [
       -20, -10, -10, -10, -10, -10, -10, -20,
       -10,   0,   0,   0,   0,   0,   0, -10,
       -10,   0,   5,  10,  10,   5,   0, -10,
       -10,   5,   5,  10,  10,   5,   5, -10,
       -10,   0,  10,  10,  10,  10,   0, -10,
       -10,  10,  10,  10,  10,  10,  10, -10,
       -10,   5,   0,   0,   0,   0,   5, -10,
       -20, -10, -10, -10, -10, -10, -10, -20
    ],
    'r': [
         0,   0,   0,   5,   5,   0,   0,   0,
        -5,   0,   0,   0,   0,   0,   0,  -5,
        -5,   0,   0,   0,   0,   0,   0,  -5,
        -5,   0,   0,   0,   0,   0,   0,  -5,
        -5,   0,   0,   0,   0,   0,   0,  -5,
        -5,   0,   0,   0,   0,   0,   0,  -5,
         5,  10,  10,  10,  10,  10,  10,   5,
         0,   0,   0,   0,   0,   0,   0,   0
    ],
    'q': [
       -20, -10, -10,  -5,  -5, -10, -10, -20,
       -10,   0,   0,   0,   0,   0,   0, -10,
       -10,   0,   5,   5,   5,   5,   0, -10,
        -5,   0,   5,   5,   5,   5,   0,  -5,
         0,   0,   5,   5,   5,   5,   0,  -5,
       -10,   5,   5,   5,   5,   5,   0, -10,
       -10,   0,   5,   0,   0,   0,   0, -10,
       -20, -10, -10,  -5,  -5, -10, -10, -20
    ],
    'k': [
       -30, -40, -40, -50, -50, -40, -40, -30,
       -30, -40, -40, -50, -50, -40, -40, -30,
       -20, -30, -30, -40, -40, -30, -30, -20,
       -10, -20, -20, -30, -30, -20, -20, -10,
         0,   0,   0,   0,   0,   0,   0,   0,
        10,  20,  20,  20,  20,  20,  20,  10,
        20,  30,  30,  30,  30,  30,  30,  20,
        20,  30,  10,   0,   0,  10,  30,  20
    ]
}

#Evaluation function for board position
def evaluate_board(board):
    score = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            value = PIECE_VALUES[piece.symbol().lower()]
            table_bonus = PIECE_SQUARE_TABLES.get(piece.symbol().lower(), [0] * 64)[square]
            mobility_bonus = len(list(board.attacks(square)))  # Add mobility as a bonus

            #Check if the piece is under attack and not defended
            under_attack = board.is_attacked_by(not piece.color, square)
            defended = board.is_attacked_by(piece.color, square)
            safety_penalty = -value * 0.5 if under_attack and not defended else 0

            score += value + (table_bonus if piece.color else -table_bonus) + safety_penalty
    return score if board.turn else -score



#########################
# Minimax with Alpha-Beta
#########################

transposition_table = defaultdict(lambda: None)

def minimax(board, depth, alpha, beta, maximizing_player):
    if depth == 0 or board.is_game_over():
        return evaluate_board(board)

    board_fen = board.fen()
    cached_value = transposition_table[board_fen]
    if cached_value is not None:
        return cached_value

    if maximizing_player:
        max_eval = float('-inf')
        for move in sorted(board.legal_moves, key=lambda m: board.is_capture(m), reverse=True):
            board.push(move)
            eval_score = minimax(board, depth - 1, alpha, beta, False)
            board.pop()
            max_eval = max(max_eval, eval_score)
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break
        transposition_table[board_fen] = max_eval
        return max_eval
    else:
        min_eval = float('inf')
        for move in sorted(board.legal_moves, key=lambda m: board.is_capture(m), reverse=True):
            board.push(move)
            eval_score = minimax(board, depth - 1, alpha, beta, True)
            board.pop()
            min_eval = min(min_eval, eval_score)
            beta = min(beta, eval_score)
            if beta <= alpha:
                break
        transposition_table[board_fen] = min_eval
        return min_eval

def softmax(values, temperature=1.0):
    """Convert values into probabilities using a numerically stable softmax."""
    max_value = max(values)  # Find the maximum value
    exp_values = [math.exp((v - max_value) / temperature) for v in values]  # Shift values
    total = sum(exp_values)
    probabilities = [v / total for v in exp_values]
    return probabilities

#This method handles the selection of the AI move, favouring the best AI move most of the time probabilistically
def get_ai_move(board, depth=3, best_move_weight=0.9):
    move_scores = {}
    for move in board.legal_moves:
        board.push(move)
        value = minimax(board, depth - 1, float('-inf'), float('inf'), not board.turn)
        board.pop()
        move_scores[move] = value

    #Find the best move
    best_move = max(move_scores, key=move_scores.get) if board.turn else min(move_scores, key=move_scores.get)

    #Add randomness for moves other than the bes move
    other_moves = [move for move in move_scores if move != best_move]
    if not other_moves:
        return best_move  # No other moves available, must choose the best move

    #Assign probabilities to the moves
    probabilities = {}
    probabilities[best_move] = best_move_weight
    other_weight = (1 - best_move_weight) / len(other_moves)
    for move in other_moves:
        probabilities[move] = other_weight

    #Select a move based on the weighted probabilities
    chosen_move = random.choices(list(probabilities.keys()), weights=list(probabilities.values()), k=1)[0]
    return chosen_move



#This method handles displaying the final message (winning, losing, anything else)
def display_end_message(screen, message):
    font = pygame.font.SysFont(None, 75)
    text = font.render(message, True, (255, 0, 0))
    text_rect = text.get_rect(center=(WIDTH // 2, HEIGHT // 2))
    screen.blit(text, text_rect)

    pygame.display.flip()
    pygame.time.wait(3000)



def main():
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Chess AI")
    clock = pygame.time.Clock()
    load_images()

    board = chess.Board()
    selected_square = None
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                col, row = x // SQUARE_SIZE, y // SQUARE_SIZE
                clicked_sq = chess.square(col, 7 - row)

                if selected_square is None:
                    selected_square = clicked_sq
                else:
                    move = chess.Move(selected_square, clicked_sq)
                    if move in board.legal_moves:
                        board.push(move)
                        selected_square = None

                        draw_board(screen)
                        draw_pieces(screen, board)
                        pygame.display.flip()

                        if board.is_game_over():
                            if board.is_checkmate():
                                display_end_message(screen, "You Lose!" if board.turn else "You Win!")
                            else:
                                display_end_message(screen, "Draw!")
                            running = False
                            break

                        time.sleep(2)  #Add a 2second delay before computer moves

                        if not board.is_game_over():
                            ai_move = get_ai_move(board, depth=5, best_move_weight=0.9)
                            if ai_move in board.legal_moves:
                                board.push(ai_move)

                                #Handling message shown after winning or losing or draw
                                if board.is_game_over():
                                    if board.is_checkmate():
                                        display_end_message(screen, "You Lose!" if board.turn else "You Win!")
                                    else:
                                        display_end_message(screen, "Draw!")
                                    running = False
                                    break
                    else:
                        selected_square = None

        draw_board(screen)
        draw_pieces(screen, board)
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    main()
