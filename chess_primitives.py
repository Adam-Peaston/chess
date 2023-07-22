import re
import numpy as np
from itertools import groupby

# Encoding whether the rook and king have moved yet in order to preserve information about whether castling is still possible.
# Could also update to include different encodings for light square versus dark square bishops.
square_states = {'empty':0,
    'my_pawn':1, 'my_passant_pawn':2,
    'my_virgin_rook':3, 'my_moved_rook':4,
    'my_knight':5, 'my_ls_bishop':6, 'my_ds_bishop':7,
    'my_queen':8, 'my_virgin_king':9, 'my_moved_king':10,
    'op_pawn':11, 'op_passant_pawn':12,
    'op_virgin_rook':13, 'op_moved_rook':14,
    'op_knight':15, 'op_ls_bishop':16, 'op_ds_bishop':17,
    'op_queen':18, 'op_virgin_king':19, 'op_moved_king':20
}

piece_points = { # piece_ind: points
    1:1, 2:1, 3:5, 4:5, 5:3, 6:3, 7:3, 8:9,
    11:1, 12:1, 13:5, 14:5, 15:3, 16:3, 17:3, 18:9,
}

# Mappers to help convert from rank/file to array index based on white/black perspective.
white_filemap = {v:k for k,v in dict(enumerate('abcdefgh')).items()}
black_filemap = {v:k for k,v in dict(enumerate('hgfedcba')).items()}
white_rankmap = {rank:8-int(rank) for rank in range(1,9)}
black_rankmap = {rank:int(rank)-1 for rank in range(1,9)}
board_map = {
    'rank': {'white': white_rankmap, 'black': black_rankmap},
    'file': {'white': white_filemap, 'black': black_filemap}
}

# Mapper from algrbraic notation to piece index
piece_map = {'P': {1, 2}, 'R': {3, 4}, 'N': {5}, 'B': {6, 7}, 'Q': {8}, 'K': {9, 10}}

def init_board(play_as='white'):
    '''Initializes board for new game.'''
    if play_as == 'white': # Queen - King
        my_pieces = ['my_virgin_rook', 'my_knight', 'my_ds_bishop', 'my_queen', 'my_virgin_king', 'my_ls_bishop', 'my_knight', 'my_virgin_rook']
        op_pieces = ['op_virgin_rook', 'op_knight', 'op_ls_bishop', 'op_queen', 'op_virgin_king', 'op_ds_bishop', 'op_knight', 'op_virgin_rook']
    elif play_as == 'black': # King - Queen
        my_pieces = ['my_virgin_rook', 'my_knight', 'my_ds_bishop', 'my_virgin_king', 'my_queen', 'my_ls_bishop', 'my_knight', 'my_virgin_rook']
        op_pieces = ['op_virgin_rook', 'op_knight', 'op_ls_bishop', 'op_virgin_king', 'op_queen', 'op_ds_bishop', 'op_knight', 'op_virgin_rook']
    board = np.zeros((8,8), dtype=int) # Initialize Board
    board[-2] = np.array([1]*8, dtype=int) # My Pawns
    board[1] = np.array([11]*8, dtype=int) # Op Pawns
    board[-1] = np.array([square_states[p] for p in my_pieces], dtype=int) # My pieces
    board[0] = np.array([square_states[p] for p in op_pieces], dtype=int) # Op pieces
    return board

def conjugate_board(board):
    '''Swaps board perspective to opponent. Will be useful for self-play.'''
    conj_board = np.flip(board)
    return np.where(conj_board >= 11, conj_board - 10, np.where(conj_board >= 1, conj_board + 10, 0))

###:: COMPUTING AVAILABLE MOVES ::###

def on_board(r, c):
    '''Returns true if given row / column is on the board.'''
    return (0 <= r <= 7) & (0 <= c <= 7)

def is_light_square(r, c):
    '''Returns true if the row / column position is a light square.'''
    # Satisfy yourself that this is invariant to player perspective.
    return ((r + c) % 2) == 0

def pawn_moves(board):
    pawn_locs = np.transpose(((board == 1) | (board == 2)).nonzero())
    candidate_boards = []
    
    # Scan for available pawn moves
    for r,c in pawn_locs:

        # Can move forward one? 
        # rp: proposed position row, cp: proposed position column
        rp, cp = r-1, c
        if on_board(rp, cp): # Proposed position is on the board
            if (board[rp, cp] == 0): # Forward square clear
                candidate = board.copy()
                candidate[rp,cp] = 1
                candidate[r,c] = 0
                candidate_boards.append(candidate)

                if rp == 0: # Option to convert to piece
                    # If we create a bishop, it is a light square or dark square bishop?
                    bishop = 6 if is_light_square(rp,cp) else 7
                    for piece in [4,5,bishop,8]:
                        candidate = board.copy()
                        candidate[rp,cp] = piece
                        candidate[r,c] = 0
                        candidate_boards.append(candidate)

        # Forward two?
        rp, cp = r-2, c
        if r == 6: # Pawn not yet moved
            if (board[rp+1, cp] == 0): # First forward square clear
                if (board[rp, cp] == 0): # Second forward square clear
                    candidate = board.copy()
                    candidate[rp,cp] = 2
                    candidate[r,c] = 0
                    candidate_boards.append(candidate)

        # Forward left?
        rp, cp = r-1, c-1
        if on_board(rp, cp): # Proposed position is on the board
            if (board[rp, cp] >= 11): # Forward left square enemy-populated
                candidate = board.copy()
                candidate[rp,cp] = 1
                candidate[r,c] = 0
                candidate_boards.append(candidate)

                if rp == 0: # Option to convert to piece
                    bishop = 6 if is_light_square(rp,cp) else 7
                    for piece in [4,5,bishop,8]:
                        candidate = board.copy()
                        candidate[rp,cp] = piece
                        candidate[r,c] = 0
                        candidate_boards.append(candidate)
                        
        # Forward-left en-passant. Needs game moderator to enforce lapsing of en-passant option.
        rp, cp = r-1, c-1
        if on_board(rp, cp):
            if (board[r, cp] == 12): # Adjacent square occupied by enemy passant pawn.
                candidate = board.copy()
                candidate[rp,cp] = 1
                candidate[r,c] = 0
                candidate[r,cp] = 0 # Enemy passant pawn captured.
                candidate_boards.append(candidate)

        # Forward right?
        rp, cp = r-1, c+1
        if on_board(rp, cp): # Proposed position is on the board
            if (board[rp, cp] >= 11): # Forward right square enemy-populated
                candidate = board.copy()
                candidate[rp,cp] = 1
                candidate[r,c] = 0
                candidate_boards.append(candidate)

                if rp == 0: # Option to convert to piece
                    bishop = 6 if is_light_square(rp,cp) else 7
                    for piece in [4,5,bishop,8]:
                        candidate = board.copy()
                        candidate[rp,cp] = piece
                        candidate[r,c] = 0
                        candidate_boards.append(candidate)
                        
        # Forward-right en-passant. Needs game moderator to enforce lapsing of en-passant option.
        rp, cp = r-1, c+1
        if on_board(rp, cp):
            if (board[r, cp] == 12): # Adjacent square occupied by enemy passant pawn.
                candidate = board.copy()
                candidate[rp,cp] = 1
                candidate[r,c] = 0
                candidate[r,cp] = 0 # Enemy passant pawn captured.
                candidate_boards.append(candidate)
                        
    return candidate_boards

def rook_moves(board):
    rook_locs = np.transpose(((board==3) | (board==4)).nonzero())
    candidate_boards = []

    for r,c in rook_locs:

        # Move left?
        offset = 1
        while True:
            rp, cp = r, c-offset
            if on_board(rp, cp):
                if board[rp, cp] == 0: # Proposed position clear
                    candidate = board.copy()
                    candidate[rp,cp] = 4 # Becomes moved rook
                    candidate[r,c] = 0
                    candidate_boards.append(candidate)
                    offset += 1 # Increment offset
                elif board[rp, cp] >= 11: # Can take opposition piece
                    candidate = board.copy()
                    candidate[rp,cp] = 4 # Becomes moved rook
                    candidate[r,c] = 0
                    candidate_boards.append(candidate)
                    break
                else:
                    break
            else:
                break

        # Move right?
        offset = 1
        while True:
            rp, cp = r, c+offset
            if on_board(rp, cp):
                if board[rp, cp] == 0: # Proposed position clear
                    candidate = board.copy()
                    candidate[rp,cp] = 4 # Becomes moved rook
                    candidate[r,c] = 0
                    candidate_boards.append(candidate)
                    offset += 1 # Increment offset
                elif board[rp, cp] >= 11: # Can take opposition piece
                    candidate = board.copy()
                    candidate[rp,cp] = 4 # Becomes moved rook
                    candidate[r,c] = 0
                    candidate_boards.append(candidate)
                    break # End search
                else:
                    break # End search
            else:
                break

        # Move forward?
        offset = 1
        while True:
            rp, cp = r-offset, c
            if on_board(rp, cp):
                if board[rp, cp] == 0: # Proposed position clear
                    candidate = board.copy()
                    candidate[rp,cp] = 4 # Becomes moved rook
                    candidate[r,c] = 0
                    candidate_boards.append(candidate)
                    offset += 1 # Increment offset
                elif board[rp, cp] >= 11: # Can take opposition piece
                    candidate = board.copy()
                    candidate[rp,cp] = 4 # Becomes moved rook
                    candidate[r,c] = 0
                    candidate_boards.append(candidate)
                    break # End search
                else:
                    break # End search
            else:
                break

        # Move backward?
        offset = 1
        while True:
            rp, cp = r+offset, c
            if on_board(rp, cp):
                if board[rp, cp] == 0: # Proposed position clear
                    candidate = board.copy()
                    candidate[rp,cp] = 4 # Becomes moved rook
                    candidate[r,c] = 0
                    candidate_boards.append(candidate)
                    offset += 1 # Increment offset
                elif board[rp, cp] >= 11: # Can take opposition piece
                    candidate = board.copy()
                    candidate[rp,cp] = 4 # Becomes moved rook
                    candidate[r,c] = 0
                    candidate_boards.append(candidate)
                    break # End search
                else:
                    break # End search
            else:
                break
    
    return candidate_boards


def castle_moves(board):
    rook_locs = np.transpose(((board==3) | (board==4)).nonzero())
    king_loc = np.transpose(((board==9) | (board==10)).nonzero())[0] # Only ever one
    kr, kc = king_loc
    
    candidate_boards = []

    for r,c in rook_locs:
        if board[r,c] == 3: # Must be rook's first move
            if board[kr,kc] == 9: # Must be king's first move
                rng = sorted([c,kc]) # Rook and King columns in sorted order
                rng = range(rng[0]+1,rng[1]) # Range over intermediate columns
                if all([board[7,ci] == 0 for ci in rng]): # Path must be clear of other pieces
                    # Is the enemy attacking any of the path squares?
                    enemy_board = conjugate_board(board) # Swap to their perspective
                    # Enemy moves considered include all but castle, which can otherwise trigger an infinite recursion. Enemy castle cannot attack back rank anyway.
                    enemy_moves = pawn_moves(enemy_board) + rook_moves(enemy_board) + knight_moves(enemy_board) + bishop_moves(enemy_board) + queen_moves(enemy_board) + king_moves(enemy_board)
                    enemy_moves = [conjugate_board(em) for em in enemy_moves] # Conjugate back to our perspective

                    if c == 0: # Castle left?
                        if all([all([enm_move[7,ci] == 0 for ci in [kc-1, kc-2]]) for enm_move in enemy_moves]):
                            candidate = board.copy()
                            candidate[7,kc-2] = 10 # Move King
                            candidate[7,kc] = 0
                            candidate[7,kc-1] = 4 # Move rook
                            candidate[7,0] = 0
                            candidate_boards.append(candidate)

                    elif c == 7: # Castle right?
                        if all([all([enm_move[7,ci] == 0 for ci in [kc+1, kc+2]]) for enm_move in enemy_moves]):
                            candidate = board.copy()
                            candidate[7,kc+2] = 10 # Move King
                            candidate[7,kc] = 0
                            candidate[7,kc+1] = 4 # Move rook
                            candidate[7,7] = 0
                            candidate_boards.append(candidate)
                            
    return candidate_boards

def knight_moves(board):
    knight_locs = np.transpose((board==5).nonzero())
    candidate_boards = []

    for r,c in knight_locs:

        proposals = [ # All the relative positions the rook could move to if allowed.
            (r+2,c+1),(r+1,c+2),
            (r-2,c+1),(r-1,c+2),
            (r+2,c-1),(r+1,c-2),
            (r-2,c-1),(r-1,c-2),
        ]

        for rp,cp in proposals:
            if on_board(rp, cp):
                if (board[rp, cp] == 0) or (board[rp, cp] >= 11): # Proposed position clear or enemy-occupied
                    candidate = board.copy()
                    candidate[rp,cp] = 5 # Move knight
                    candidate[r,c] = 0
                    candidate_boards.append(candidate)

    return candidate_boards

def bishop_moves(board):
    bishop_locs = np.transpose(((board==6) | (board==7)).nonzero())
    candidate_boards = []

    for r,c in bishop_locs:
        bishop = 6 if is_light_square(r,c) else 7 # Which one?

        # Move forward-left?
        offset = 1
        while True:
            rp, cp = r-offset, c-offset
            if on_board(rp, cp):
                if board[rp, cp] == 0: # Proposed position clear
                    candidate = board.copy()
                    candidate[rp,cp] = bishop # Move bishop
                    candidate[r,c] = 0
                    candidate_boards.append(candidate)
                    offset += 1 # Increment offset
                elif board[rp, cp] >= 11: # Can take opposition piece
                    candidate = board.copy()
                    candidate[rp,cp] = bishop # Move bishop
                    candidate[r,c] = 0
                    candidate_boards.append(candidate)
                    break # End search
                else:
                    break # End search
            else:
                break

        # Move forward-right?
        offset = 1
        while True:
            rp, cp = r-offset, c+offset
            if on_board(rp, cp):
                if board[rp, cp] == 0: # Proposed position clear
                    candidate = board.copy()
                    candidate[rp,cp] = bishop # Move bishop
                    candidate[r,c] = 0
                    candidate_boards.append(candidate)
                    offset += 1 # Increment offset
                elif board[rp, cp] >= 11: # Can take opposition piece
                    candidate = board.copy()
                    candidate[rp,cp] = bishop # Move bishop
                    candidate[r,c] = 0
                    candidate_boards.append(candidate)
                    break # End search
                else:
                    break # End search
            else:
                break

        # Move backward-left?
        offset = 1
        while True:
            rp, cp = r+offset, c-offset
            if on_board(rp, cp):
                if board[rp, cp] == 0: # Proposed position clear
                    candidate = board.copy()
                    candidate[rp,cp] = bishop # Move bishop
                    candidate[r,c] = 0
                    candidate_boards.append(candidate)
                    offset += 1 # Increment offset
                elif board[rp, cp] >= 11: # Can take opposition piece
                    candidate = board.copy()
                    candidate[rp,cp] = bishop # Move bishop
                    candidate[r,c] = 0
                    candidate_boards.append(candidate)
                    break # End search
                else:
                    break # End search
            else:
                break

        # Move backward-right?
        offset = 1
        while True:
            rp, cp = r+offset, c+offset
            if on_board(rp, cp):
                if board[rp, cp] == 0: # Proposed position clear
                    candidate = board.copy()
                    candidate[rp,cp] = bishop # Move bishop
                    candidate[r,c] = 0
                    candidate_boards.append(candidate)
                    offset += 1 # Increment offset
                elif board[rp, cp] >= 11: # Can take opposition piece
                    candidate = board.copy()
                    candidate[rp,cp] = bishop # Move bishop
                    candidate[r,c] = 0
                    candidate_boards.append(candidate)
                    break # End search
                else:
                    break # End search
            else:
                break
                
    return candidate_boards

def queen_moves(board):
    queen_locs = np.transpose((board==8).nonzero())
    candidate_boards = []
    
    for r,c in queen_locs:

        # Move left?
        offset = 1
        while True:
            rp, cp = r, c-offset
            if on_board(rp, cp):
                if board[rp, cp] == 0: # Proposed position clear
                    candidate = board.copy()
                    candidate[rp,cp] = 8 # Move queen
                    candidate[r,c] = 0
                    candidate_boards.append(candidate)
                    offset += 1 # Increment offset
                elif board[rp, cp] >= 11: # Can take piece
                    candidate = board.copy()
                    candidate[rp,cp] = 8 # Move queen
                    candidate[r,c] = 0
                    candidate_boards.append(candidate)
                    break # End search
                else:
                    break # End search
            else:
                break

        # Move right?
        offset = 1
        while True:
            rp, cp = r, c+offset
            if on_board(rp, cp):
                if board[rp, cp] == 0: # Proposed position clear
                    candidate = board.copy()
                    candidate[rp,cp] = 8 # Move queen
                    candidate[r,c] = 0
                    candidate_boards.append(candidate)
                    offset += 1 # Increment offset
                elif board[rp, cp] >= 11: # Can take piece
                    candidate = board.copy()
                    candidate[rp,cp] = 8 # Move queen
                    candidate[r,c] = 0
                    candidate_boards.append(candidate)
                    break # End search
                else:
                    break # End search
            else:
                break # End search

        # Move forward?
        offset = 1
        while True:
            rp, cp = r-offset, c
            if on_board(rp, cp):
                if board[rp, cp] == 0: # Proposed position clear
                    candidate = board.copy()
                    candidate[rp,cp] = 8 # Move queen
                    candidate[r,c] = 0
                    candidate_boards.append(candidate)
                    offset += 1 # Increment offset
                elif board[rp, cp] >= 11: # Can take opposition piece
                    candidate = board.copy()
                    candidate[rp,cp] = 8 # Move queen
                    candidate[r,c] = 0
                    candidate_boards.append(candidate)
                    break # End search
                else:
                    break # End search
            else:
                break # End search

        # Move backward?
        offset = 1
        while True:
            rp, cp = r+offset, c
            if on_board(rp, cp):
                if board[rp, cp] == 0: # Proposed position clear
                    candidate = board.copy()
                    candidate[rp,cp] = 8 # Move queen
                    candidate[r,c] = 0
                    candidate_boards.append(candidate)
                    offset += 1 # Increment offset
                elif board[rp, cp] >= 11: # Can take opposition piece
                    candidate = board.copy()
                    candidate[rp,cp] = 8 # Move queen
                    candidate[r,c] = 0
                    candidate_boards.append(candidate)
                    break # End search
                else:
                    break # End search
            else:
                break # End search
                
        # Move forward-left?
        offset = 1
        while True:
            rp, cp = r-offset, c-offset
            if on_board(rp, cp):
                if board[rp, cp] == 0: # Proposed position clear
                    candidate = board.copy()
                    candidate[rp,cp] = 8 # Move queen
                    candidate[r,c] = 0
                    candidate_boards.append(candidate)
                    offset += 1 # Increment offset
                elif board[rp, cp] >= 11: # Can take opposition piece
                    candidate = board.copy()
                    candidate[rp,cp] = 8 # Move queen
                    candidate[r,c] = 0
                    candidate_boards.append(candidate)
                    break # End search
                else:
                    break # End search
            else:
                break # End search

        # Move forward-right?
        offset = 1
        while True:
            rp, cp = r-offset, c+offset
            if on_board(rp, cp):
                if board[rp, cp] == 0: # Proposed position clear
                    candidate = board.copy()
                    candidate[rp,cp] = 8 # Move queen
                    candidate[r,c] = 0
                    candidate_boards.append(candidate)
                    offset += 1 # Increment offset
                elif board[rp, cp] >= 11: # Can take opposition piece
                    candidate = board.copy()
                    candidate[rp,cp] = 8 # Move queen
                    candidate[r,c] = 0
                    candidate_boards.append(candidate)
                    break # End search
                else:
                    break # End search
            else:
                break # End search

        # Move backward-left?
        offset = 1
        while True:
            rp, cp = r+offset, c-offset
            if on_board(rp, cp):
                if board[rp, cp] == 0: # Proposed position clear
                    candidate = board.copy()
                    candidate[rp,cp] = 8 # Move queen
                    candidate[r,c] = 0
                    candidate_boards.append(candidate)
                    offset += 1 # Increment offset
                elif board[rp, cp] >= 11: # Can take opposition piece
                    candidate = board.copy()
                    candidate[rp,cp] = 8 # Move queen
                    candidate[r,c] = 0
                    candidate_boards.append(candidate)
                    break # End search
                else:
                    break # End search
            else:
                break # End search

        # Move backward-right?
        offset = 1
        while True:
            rp, cp = r+offset, c+offset
            if on_board(rp, cp):
                if board[rp, cp] == 0: # Proposed position clear
                    candidate = board.copy()
                    candidate[rp,cp] = 8 # Move queen
                    candidate[r,c] = 0
                    candidate_boards.append(candidate)
                    offset += 1 # Increment offset
                elif board[rp, cp] >= 11: # Can take opposition piece
                    candidate = board.copy()
                    candidate[rp,cp] = 8 # Move queen
                    candidate[r,c] = 0
                    candidate_boards.append(candidate)
                    break # End search
                else:
                    break # End search
            else:
                break # End search
                
    return candidate_boards

def king_moves(board):
    king_loc = np.transpose(((board==9) | (board==10)).nonzero())[0] # Only ever one
    candidate_boards = []
    
    r,c = king_loc
    proposals = [
        (r+1,c),(r-1,c),
        (r,c+1),(r,c-1),
        (r+1,c+1),(r-1,c-1),
        (r+1,c-1),(r-1,c+1),
    ]
    
    for rp,cp in proposals:
        if on_board(rp, cp):
            if (board[rp, cp] == 0) or (board[rp, cp] >= 11): # Proposed position clear or enemy-occupied
                candidate = board.copy()
                candidate[rp,cp] = 10 # Move King
                candidate[r,c] = 0
                candidate_boards.append(candidate)
                
    return candidate_boards

def candidate_moves(board):
    '''Return candidate boards representing each potential move from a given board configuration.'''
    candidates = pawn_moves(board) + rook_moves(board) + knight_moves(board) + bishop_moves(board) + queen_moves(board) + king_moves(board) + castle_moves(board)
    return [c for c in candidates if not in_check(c)]

def in_check(board):
    '''Returns true if I am in check. Can then filter available moves for those that get me out of check. If none, checkmate.'''
    # Where is my King?
    king_loc = np.transpose(((board==9) | (board==10)).nonzero())[0] # Only ever one
    # Is the enemy attacking any of the path squares?
    enemy_board = conjugate_board(board) # Swap to their perspective
    # Enemy moves considered include all but castle, which can otherwise trigger an infinite recursion. Enemy castle cannot attack king anyway.
    enemy_moves = pawn_moves(enemy_board) + rook_moves(enemy_board) + knight_moves(enemy_board) + bishop_moves(enemy_board) + queen_moves(enemy_board) + king_moves(enemy_board)
    enemy_moves = [conjugate_board(em) for em in enemy_moves] # Conjugate back to our perspective
    # Not the case that where my king is currently located, the piece in that location is no longer my king, for possible enemy moves.
    # return all([((em==9)|(em==10)).sum() == 1 for em in enemy_moves]) # i.e. my king always survives my enemy's next move.
    return any([em[tuple(king_loc)] not in piece_map['K'] for em in enemy_moves]) # i.e. none of my enemy's moves can replace my king.

def is_draw(board):
    '''Returns true if board state is a known draw due to insufficient material for check-mate.'''
    # King - King
    K_K = set(board[np.where(board>0)]).issubset({9,10,19,20})
    # King - King | Knight
    K_KN = set(board[np.where(board>0)]).issubset({9,10,19,20,5})
    # King - King | Bishop
    K_KB = (
    set(board[np.where(board>0)]).issubset({9,10,19,20,6}) or
    set(board[np.where(board>0)]).issubset({9,10,19,20,7}) 
    )
    # King | Bishop - King | Bishop (bishops on opposite colors) 
    KB_KB = (
        set(board[np.where(board>0)]).issubset({9,10,19,20,6,17}) or
        set(board[np.where(board>0)]).issubset({9,10,19,20,7,16})
    )
    return (K_K or K_KN or K_KB or KB_KB)

def points_balance(board):
    # based on sum of piece points left each side - kings have no piece value. Function returns score in range (-1, 1).
    player_points = sum([(board==p).sum()*piece_points[p] for p in range(1,9)]) # Player pieces are 1 - 9
    # opponent_points = sum([(board==p).sum()*piece_points[p] for p in range(11,19)]) # Opponent pieces are 11 - 19
    total_points = player_points + sum([(board==p).sum()*piece_points[p] for p in range(11,19)]) + 1e-8 # Opponent pieces are 11 - 19
    player_score = (2 * player_points / total_points) - 1 # Adding a epsilon for numeric stability
    opponent_score = - player_score
    return player_score, opponent_score

def check_gameover(board, n_candidates, n_moves, turn, max_moves=np.inf, points_eval=False):
    # If there are no candidate moves available, that could mean checkmate or stalemate.
    if n_candidates == 0:
        if in_check(board): # I am in check and have no moves available. Checkmate, opponent wins.
            scores = (-1.0, 1.0) # Current player loses, opponent wins; (-1, 1)
            return scores # return game results
        else: # I am not in check but have no availabe moves. Stalemate. Draw.
            scores = (0.0, 0.0)
            return scores # return game results
    elif is_draw(board): # Draw.
        scores = (0.0, 0.0)
        return scores # return game results
    elif n_moves >= max_moves: # max moves reached
        if points_eval: # partial eval requested
            return points_balance(board)
        else:
            return (0.0, 0.0)
    else: # Return False, game is not over. 
        return False
    
def compress(string):
    ''' Generating short unique string tokens to represent board states.'''
    return re.sub(r'(?<![0-9])[1](?![0-9])', '', ''.join('%s%s' % (char, sum(1 for _ in group)) for char, group in groupby(string)))

def board_token(board, prefix):
    body = ''.join(list(map(lambda n: dict(enumerate('abcdefghijklmnopqrstuvwxyz'))[n], board.flatten().tolist())))
    return prefix + compress(body)

def token_board(token, splitter='_'):
    bstring = token.split(splitter)[1]
    blist = list(bstring)
    reverse = {j:i for i,j in dict(enumerate('abcdefghijklmnopqrstuvwxyz')).items()}
    board = []
    char_buff = ''
    int_buff = ''
    for c in blist:
        if c.isalpha(): # The character is a letter
            if int_buff != '': # There is a character in the buffer and we've just finished working out how many.
                m = int(int_buff) # How many of the previous token to generate
                board += [char_buff]*m # Add them to the board list
                int_buff = ''
                char_buff = reverse[c] # Set the char_buff equal to the new token
            else: # There was no multiple of the previous character to apply
                if char_buff != '':
                    board += [char_buff] # Add the previous token to the board
                char_buff = reverse[c] # Set the char_buff equal to the new token
        else:
            int_buff += c
    if int_buff != '': # There is a character in the buffer and we've just finished working out how many.
        m = int(int_buff) # How many of the previous token to generate
        board += [char_buff]*m # Add them to the board list
    else:
        board += [char_buff] # Add the previous token to the board
    return np.array(board).reshape(8,8)

def play_historic(board, moves, turn):
    '''
    Wrote this function to play historic games based on the sequence of moves to verify that all moves made are legal in the above framework.
    '''
    
    # Output of function
    states = []
    
    # Loop over provided moves
    for move in moves:

        # Retrieve the move
        move = move.replace('x','').replace('+','').replace('#','') # Drop capture/check/checkmate notation - unnecessary
        castle_dest = None
        promote_to = None

        # Revert any passant pawns to regular pawns if they survived the enemy's turn.
        board[board==2] = 1

        # Detect pawn promotion
        if '=' in move:
            promote_to = move[-1]
            move = move[:-2]

        # A few special cases to handle
        if move == '1-0':
            print('White wins.')
            break

        elif move == '0-1':
            print('Black wins.')
            break

        elif move == '1/2-1/2':
            print('Draw.')
            break

        elif move == 'O-O': # Castle king side
            if turn == 'white':
                king_dest = (7,6)
                rook_dest = (7,5)
                castle_dest = [king_dest, rook_dest]
            elif turn == 'black':
                king_dest = (7,1)
                rook_dest = (7,2)
                castle_dest = [king_dest, rook_dest]
            else:
                print('Failed to resolve turn for king-side castle.')

        elif move == 'O-O-O': # Castle queen side
            if turn == 'white':
                king_dest = (7,2)
                rook_dest = (7,3)
                castle_dest = [king_dest, rook_dest]
            elif turn == 'black':
                king_dest = (7,5)
                rook_dest = (7,4)
                castle_dest = [king_dest, rook_dest]

        # Piece move - analyse the move description for piece origin and destination information
        else:
            if move[0].islower():
                piece = 'P'
                orig = move[:-2]
            else:
                piece = move[0]
                orig = move[1:-2]

            # Destination
            dest = move[-2:]
            file, rank = dest
            r_dest = board_map['rank'][turn][int(rank)]
            c_dest = board_map['file'][turn][file]

            # Origin if specified
            if len(orig) == 0:
                r_orig = None
                c_orig = None

            elif len(orig) == 1:

                if orig.isnumeric():
                    rank = int(orig)
                    r_orig = board_map['rank'][turn][rank]
                    c_orig = None
                else:
                    file = orig
                    c_orig = board_map['file'][turn][file]
                    r_orig = None

            elif len(orig) == 2:

                file, rank = origin
                r_orig = board_map['rank'][turn][rank]
                c_orig = board_map['file'][turn][file]

        # Calculate legal moves from this position
        candidates = candidate_moves(board)

        # Filter boards based on king/rook destinations
        if castle_dest: # Castle dest not None only if move passed in was a castle move.
            king_dest, rook_dest = castle_dest
            played = [c for c in candidates if c[king_dest]==10 and c[rook_dest]==4]

        else:
            # Filter boards based on destination
            if promote_to:
                played = [c for c in candidates if c[r_dest, c_dest] in piece_map[promote_to]]
            else:
                played = [c for c in candidates if c[r_dest, c_dest] in piece_map[piece]]

            if piece == 'R': # Also check here that if it's a rook moving, that the king hasn't moved; not a castle move.
                king_loc = np.transpose(((board==9) | (board==10)).nonzero())[0] # Only ever one
                played = [c for c in played if c[tuple(king_loc)] == board[tuple(king_loc)]]

            # Filter boards based on origin
            if len(played) > 1:
                if not str(r_orig).isnumeric(): # Origin row implied
                    r_orig = np.concatenate([list((board[:,c_orig]==i).nonzero()) for i in piece_map[piece]], axis=1).item()
                elif not str(c_orig).isnumeric(): # Origin column implied
                    c_orig = np.concatenate([list((board[r_orig,:]==i).nonzero()) for i in piece_map[piece]], axis=1).item()
                else: # Origin fully specified
                    assert str(r_orig).isnumeric() and str(c_orig).isnumeric() # Origin fully specified
                played = [c for c in played if c[r_orig, c_orig] == 0]

        # Logic above should have identified the one move that was played.
        assert len(played) == 1
        played = played[0]
        
        # Record the new board state
        board = played
        
        # Save candidates and played. Maybe also save elo to be able to train a model that is parameterised by strength?
        states.append((turn, board, candidates))
        
        # Conjugate board to other side's view
        board = conjugate_board(board)

        # Turn play over to the other side.
        if turn == 'white':
            turn = 'black'
        else:
            turn = 'white'
        
    return states