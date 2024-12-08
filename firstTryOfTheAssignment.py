from numba import njit, prange
import numpy as np
import time

# Initialize the game board
def initialize_board(N):
    return np.zeros((N, N), dtype=int)

# Check if a move is valid
def is_valid_move(board, row, col):
    return board[row, col] == 0

# Check for a win
@njit
def check_win(board, player):
    N = board.shape[0]
    # Check rows and columns
    for i in range(N):
        if np.all(board[i, :] == player) or np.all(board[:, i] == player):
            return True
    # Check diagonals
    if np.all(np.diag(board) == player) or np.all(np.diag(np.fliplr(board)) == player):
        return True
    return False

# Parallelized function for computer move
@njit(parallel=True)
def computer_move(board, player):
    N = board.shape[0]
    best_move = (-1, -1)
    for i in prange(N):
        for j in range(N):
            if board[i, j] == 0:
                board[i, j] = player
                if check_win(board, player):
                    best_move = (i, j)
                board[i, j] = 0
    return best_move


# Player vs Computer game loop
@njit(parallel=True)
def computer_move(board, player):
    N = board.shape[0]
    best_move = (-1, -1)
    for i in prange(N):
        for j in range(N):
            if board[i, j] == 0:
                board[i, j] = player
                if check_win(board, player):
                    best_move = (i, j)
                board[i, j] = 0
    return best_move

# Player vs Computer game loop
def player_vs_computer(N):
    board = initialize_board(N)
    player = 1
    computer = 2
    while True:
        # Player move
        row, col = map(int, input("Enter your move (row col): ").split())
        if is_valid_move(board, row, col):
            board[row, col] = player
            if check_win(board, player):
                print("Player wins!")
                break
        else:
            print("Invalid move. Try again.")
            continue
        # Computer move
        start = time.time()
        row, col = computer_move(board, computer)
        if row != -1:
            board[row, col] = computer
        print(f"Computer move: {row}, {col}, Time: {time.time() - start} seconds")
        if row == -1:
            print("Draw!")
            break
        if check_win(board, computer):
            print("Computer wins!")
            break
        print(board)



# Computer vs Computer game loop
def computer_vs_computer(N, debug=False):
    board = initialize_board(N)
    player1 = 1
    player2 = 2
    while True:
        # Player 1 move
        row, col = computer_move(board, player1)
        if row != -1:
            board[row, col] = player1
        if row == -1:
            if debug:
                print(board)
            break
        if check_win(board, player1):
            if debug:
                print(board)
            break
        if debug:
            print(board)
        # Player 2 move
        row, col = computer_move(board, player2)
        if row != -1:
            board[row, col] = player2
        if row == -1:
            if debug:
                print(board)
            break
        if check_win(board, player2):
            if debug:
                print(board)
            break
        if debug:
            print(board)

# Main execution
if __name__ == "__main__":
    N = 3  # Change this to test different board sizes
    player_vs_computer(N)
    computer_vs_computer(N, debug=True)
