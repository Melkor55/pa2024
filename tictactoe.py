from numba import njit, prange, config, threading_layer
import numpy as np
import time

# Set the number of threads 
config.NUMBA_NUM_THREADS = 1
# Set order of preference for threading layers
config.THREADING_LAYER_PRIORITY = ["omp", "tbb", "workqueue"]
# set the threading layer before any parallel target compilation
config.THREADING_LAYER = 'threadsafe'

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
    for i in range(N):
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
    best_move = np.array([-1, -1], dtype=np.int64)  # Use Numba-friendly array
    found_win = False
    for i in prange(N):
        for j in prange(N):
            if board[i, j] == 0 and not found_win:
                board[i, j] = player
                if check_win(board, player):
                    best_move[0], best_move[1] = i, j
                    found_win = True
                board[i, j] = 0
                
                if best_move[0] == -1 and best_move[1] == -1:
                    best_move[0], best_move[1] = i, j

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
        # computer_move.parallel_diagnostics(level=4)
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
def computer_vs_computer(N, debug_mode):
    board = initialize_board(N)
    player1 = 1
    player2 = 2
    total_time = 0
    while True:
        # Player 1 move
        start = time.perf_counter()
        row, col = computer_move(board, player1)
        # computer_move.parallel_diagnostics(level=4)
        end = time.perf_counter()
        if row != -1:
            board[row, col] = player1
            _time = end - start
            total_time += _time
        print(f"Player 1 move: {row}, {col}, Time: {_time:.10f} seconds")
        if row == -1:
            print("Draw!")
            print(board)
            break
        if check_win(board, player1):
            print("Player 1 wins!")
            print(board)
            break
        if debug_mode:
            print(board)

        # Player 2 move
        start = time.perf_counter()
        row, col = computer_move(board, player2)
        # computer_move.parallel_diagnostics(level=4)
        end = time.perf_counter()
        if row != -1:
            board[row, col] = player2
            _time = end - start
            total_time += _time
        print(f"Player 2 move: {row}, {col}, Time: {_time:.10f} seconds")
        if row == -1:
            print("Draw!")
            print(board)
            break
        if check_win(board, player2):
            print("Player 2 wins!")
            print(board)
            break
        if debug_mode:
            print(board)
    print(f"Total time: {total_time:.10f} seconds")

# Main execution
if __name__ == "__main__":
    N = 10
    # player_vs_computer(N)
    debug_mode = False
    computer_vs_computer(N, debug_mode)
    print(f"Program is using {config.NUMBA_NUM_THREADS} threads")
    # demonstrate the threading layer chosen
    print("Threading layer chosen: %s" % threading_layer())