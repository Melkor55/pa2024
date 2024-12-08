from numba import njit, prange, config
import numpy as np
import time

# Set the number of threads 
config.NUMBA_NUM_THREADS = 2

# Initialize the game board
def initialize_board(N):
    return np.zeros((N, N), dtype=int)

# Check if a move is valid
def is_valid_move(board, row, col):
    return 0 <= row < board.shape[0] and 0 <= col < board.shape[1] and board[row, col] == 0

# Check for a win
@njit
def check_win(board, player):
    N = board.shape[0]
    # Check rows and columns
    for i in prange(N):
        if np.all(board[i, :] == player) or np.all(board[:, i] == player):
            return True
    # Check diagonals
    if np.all(np.diag(board) == player) or np.all(np.diag(np.fliplr(board)) == player):
        return True
    return False

# Function for computer move
@njit(parallel=True)
def computer_move(board, player):
    N = board.shape[0]
    best_move = (-1, -1)
    for i in prange(N):
        for j in prange(N):
            if board[i, j] == 0:
                board[i, j] = player
                if check_win(board, player):
                    return i, j
                board[i, j] = 0
                if best_move == (-1, -1):
                    best_move = (i, j)
    return best_move

# Player vs Computer game loop
def player_vs_computer(N):
    board = initialize_board(N)
    player = 1
    computer = 2
    while True:
        # Player move
        try:
            row, col = map(int, input("Enter your move (row col): ").split())
            if not (0 <= row < N and 0 <= col < N):
                raise ValueError
        except ValueError:
            print(f"Invalid input. Please enter two numbers between 0 and {N-1}.")
            continue

        if is_valid_move(board, row, col):
            board[row, col] = player
            if check_win(board, player):
                print("Player wins!")
                print(board)
                break
        else:
            print("Invalid move. Try again.")
            continue

        # Computer move
        start = time.perf_counter()
        row, col = computer_move(board, computer)
        computer_move.parallel_diagnostics(level=4)
        end = time.perf_counter()
        if row != -1:
            board[row, col] = computer
        print(f"Computer move: {row}, {col}, Time: {end - start:.10f} seconds")
        if row == -1:
            print("Draw!")
            print(board)
            break
        if check_win(board, computer):
            print("Computer wins!")
            print(board)
            break
        print(board)

# Computer vs Computer game loop
def computer_vs_computer(N):
    board = initialize_board(N)
    player1 = 1
    player2 = 2
    while True:
        # Player 1 move
        start = time.perf_counter()
        row, col = computer_move(board, player1)
        computer_move.parallel_diagnostics(level=4)
        end = time.perf_counter()
        if row != -1:
            board[row, col] = player1
        print(f"Player 1 move: {row}, {col}, Time: {end - start:.10f} seconds")
        if row == -1:
            print("Draw!")
            print(board)
            break
        if check_win(board, player1):
            print("Player 1 wins!")
            print(board)
            break
        print(board)

        # Player 2 move
        start = time.perf_counter()
        row, col = computer_move(board, player2)
        computer_move.parallel_diagnostics(level=4)
        end = time.perf_counter()
        if row != -1:
            board[row, col] = player2
        print(f"Player 2 move: {row}, {col}, Time: {end - start:.10f} seconds")
        if row == -1:
            print("Draw!")
            print(board)
            break
        if check_win(board, player2):
            print("Player 2 wins!")
            print(board)
            break
        print(board)

# Main execution
if __name__ == "__main__":
    print(f"Program is using {config.NUMBA_NUM_THREADS} threads")
    N = 5
    # player_vs_computer(N)
    computer_vs_computer(N)
