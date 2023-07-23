import numpy as np
import os, pickle, uuid, torch
from random import choices
from itertools import accumulate
from functools import partial
from multiprocessing import Pool
from chess_primitives import init_board, board_token, conjugate_board, candidate_moves, in_check, is_draw
from chess_model import PiecesModel, ChessAI, TransformerModel

def softmax_temp(x, temp=1):
    z = np.exp((x - x.max()) / temp)
    return z / z.sum()

def entropy(d):
    # Returns the entropy of a discrete distribution
    e = -(d * np.log2(d + 1e-10)).sum() # epsilon value added due to log2(0) == undefined.
    return e

def entropy_temperature(x, target_entropy, T=[1e-3, 1e0, 1e2], tol=1e-3, max_iter=10):
    # returns the temperature parameter required to transform the vector x into a probability distribution with a particular target entropy
    delta = np.inf
    for _ in range(max_iter):
        if delta > tol:
            E = [entropy(softmax_temp(x, temp=t)) for t in T]
            if E[0] > target_entropy:
                T = [T[0]/2, T[1], T[2]]
            elif E[2] < target_entropy:
                T = [T[0], T[1], T[2]*2]
            elif E[0] < target_entropy < E[1]:
                T = [T[0], (T[0]+T[1])/2, T[1]]
            elif E[1] < target_entropy < E[2]:
                T = [T[1], (T[1]+T[2])/2, T[2]]
            delta = T[2] - T[1]
        else:
            return (T[0]+T[2]) / 2
    return (T[0]+T[2]) / 2


def play_tournament(agents, num_games, starting_state=None, max_moves=float('inf')):
    # plays a number of paired games, one with agent0 as white, the other with agent0 as black.
    tournament_results = dict()
    for game in range(num_games):
        tournament_results[game] = dict()
        # play game with FIRST model as white
        kwargs = {'agents': {'white': agents[0], 'black': agents[1]}, 'starting_state': starting_state, 'max_moves': max_moves}
        game_result = play_game(**kwargs)
        # game_results: {'white': {'moves': [(end_token, end_board), (end_token, end_board), ...], 'points': float}, 'black': {...}}
        tournament_results[game]['a0wa1b'] = game_result
        # play game with SECOND model as white
        kwargs = {'agents': {'white': agents[1], 'black': agents[0]}, 'starting_state': starting_state, 'max_moves': max_moves}
        game_result = play_game(**kwargs)
        # game_results: {'white': {'moves': [(end_token, end_board), (end_token, end_board), ...], 'points': float}, 'black': {...}}
        tournament_results[game]['a1wa0b'] = game_result
    return tournament_results

def play_game(agents, starting_state=None, max_moves=float('inf'), verbose=False):

    board, color_toplay = starting_state if starting_state is not None else (init_board(play_as='white'), 'white')
    game_result = {'white': {'moves': [], 'points': 0}, 'black': {'moves': [], 'points': 0}}

    # Play a game until game over.
    while True:

        # Revert any passant pawns to regular pawns if they survived the last turn.
        board[board == 2] = 1
        # Options from each of the starting positions - init as empty dict
        options = {board_token(cand_board, f'{color_toplay}E_'):cand_board for cand_board in candidate_moves(board)}

        # Check if checkmate or draw.
        player_points, opponent_points, outcome = (None, None, None)
        if len(options) == 0:
            if in_check(board): # Checkmate
                player_points, opponent_points = (-1.0, 1.0)
                outcome = 'Checkmate'
            else: # Stalemate
                player_points, opponent_points = (0.0, 0.0)
                outcome = 'Stalemate, draw, or timeout'
        if is_draw(board) or len(game_result[color_toplay]['moves']) >= max_moves: # Known draw or max moves reached
            player_points, opponent_points = (0.0, 0.0)
            outcome = 'Stalemate, draw, or timeout'
        if player_points is not None:
            player, opponent = ('white', 'black') if color_toplay == 'white' else ('black','white')
            game_result[player]['points'] = player_points
            game_result[opponent]['points'] = opponent_points
            if verbose:
                print(f"{outcome} after {len(game_result[color_toplay]['moves'])} moves.")
            return game_result

        # Select end_token
        kwargs = {'options': options, 'starting_state': (board, color_toplay)}
        selected_end_token = agents[color_toplay].select_move(**kwargs)
        selected_end_board = options[selected_end_token]

        # Add this move to the game_record
        game_result[color_toplay]['moves'].append((selected_end_token, selected_end_board))

        # Swap to opponent's perspective
        color_toplay = 'white' if color_toplay == 'black' else 'black' # Swap to my turn
        board = conjugate_board(selected_end_board) # Conjugate selected_end_board to opponents perspective

class Agent:
    def __init__(self, model=None, num_simgames=10, max_simmoves=5, C=1, p=0.3, k=3):
        self.model, self.num_simgames, self.max_simmoves, self.C, self.p, self.k = model, num_simgames, max_simmoves, C, p, k

    def select_move(self, options, starting_state=None):

        # If there is no model passed, then just chose randomly.
        if self.model is None:
            return choices(list(options.keys()))[0]
        
        # Unpack options dictionary {end_token: board, ...}
        end_tokens, end_boards = zip(*options.items())

        # Obtain model scores :: predicted points at the end of game from each board; scores in the range [-1, 1]
        x = np.stack(end_boards, axis=0).reshape(-1, 64) # (board, rank, file) --> (board, square)
        model_scores = self.model.evaluate(x).flatten()
        torch.cuda.empty_cache()

        # Move selection is determined according to both model scores and simulated scores.
        sim_results = self.sim_tournament(starting_state)
        sim_summary = summarise_sim_tournament(sim_results)

        # Estimated win probabilities from simulaiton - Q analog. Zero for options not explored in the simulation.
        points = np.array([sim_summary.get(end_token, {'points': 0})['points'] for end_token in end_tokens])
        visits = np.array([sim_summary.get(end_token, {'visits': 1})['visits'] for end_token in end_tokens])
        simulation_scores = points / visits

        # Simpler than alpha zero policy equation;
        # At C=0, only consider model_scores which is noisy. At C=inf consider only simulation_scores which is best guess.
        selection_scores = model_scores + self.C * simulation_scores
        
        # Select end token
        selected_end_token = self.selector(end_tokens, selection_scores, self.p, self.k)

        return selected_end_token

    def sim_tournament(self, starting_state=None):

        # starting_state = (board, color_toplay)
        start_board, color_toplay = starting_state if starting_state is not None else (init_board(play_as='white'), 'white')
        game_boards = {game: start_board for game in range(self.num_simgames)}
        game_tokens = {game: board_token(board, f'{color_toplay}S_') for game,board in game_boards.items()}
        tournament_results = {game: {'white': {'moves': [], 'points': 0}, 'black': {'moves': [], 'points': 0}} for game in game_boards}

        # Play games in parallel until all games are over.
        while True:

            # Revert any passant pawns to regular pawns if they survived the last turn.
            game_boards = {game: np.where(board==2, 1, board) for game,board in game_boards.items()}
            game_tokens = {game: board_token(board, f'{color_toplay}S_') for game,board in game_boards.items()}
            start_tokens_boards = {game_tokens[game]: game_boards[game] for game in game_boards}

            # Options from each of the starting positions - init as empty dict
            options = {start_token: {board_token(cand, f'{color_toplay}E_'):cand for cand in candidate_moves(board)} for start_token,board in start_tokens_boards.items()}
            
            # handle checkmate, stalemate, or known draw.
            no_options = [game for game in game_boards if len(options[game_tokens[game]]) == 0]
            checkmates = [game for game in game_boards if in_check(game_boards[game]) and game in no_options]
            stalemates = [game for game in game_boards if game not in checkmates and game in no_options]
            known_draws = [game for game in game_boards if is_draw(game_boards[game])]

            for game in checkmates:
                player_points, opponent_points = (-1.0, 1.0) # Current player loses, opponent wins; (-1, 1)
            for game in known_draws + stalemates:
                player_points, opponent_points = (0.0, 0.0)
            for game in checkmates + known_draws + stalemates:
                player, opponent = ('white', 'black') if color_toplay == 'white' else ('black','white')
                tournament_results[game][player]['points'] = player_points
                tournament_results[game][opponent]['points'] = opponent_points
                del game_boards[game] # Terminate game
                del game_tokens[game] # Terminate game
                if len(game_boards) == 0: # Terminate tournament
                    return tournament_results
                
            # Unify the options dictionaries into one dict of {end_token: board, ...}
            end_tokens_boards = dict()
            for opt in options.values():
                end_tokens_boards.update(opt)
            end_tokens, end_boards = zip(*end_tokens_boards.items())

            # Obtain predicted points at the end of game from each board, scores in the range [-1, 1]
            x = np.stack(end_boards, axis=0).reshape(-1, 64) # (board, rank, file) --> (board, square)
            scores = self.model.evaluate(x).flatten()
            torch.cuda.empty_cache()

            # Complete set of model scores for all options
            option_scores = {end_tokens:score for end_tokens,score in zip(end_tokens,scores)}

            # Now expand back out to ensure options shared by parallel games appear in each game's score groups
            game_end_tokens = {game:tuple(options[game_tokens[game]].keys()) for game in game_boards} # [(end_token, end_token, ...), (...), ...]
            game_end_scores = {game:np.array([option_scores[token] for token in game_end_tokens[game]]) for game in game_boards}

            # Handle n_moves = max_moves; use model evaluations as guess for game evaluation
            exhausted = [game for game in game_boards if len(tournament_results[game][color_toplay]['moves']) >= self.max_simmoves]
            for game in exhausted:
                player_points = game_end_scores[game].max() # best of the board evaluations available at the final move
                player, opponent = ('white', 'black') if color_toplay == 'white' else ('black','white')
                opponent_points = - player_points
                tournament_results[game][player]['points'] = player_points
                tournament_results[game][opponent]['points'] = opponent_points
                del game_boards[game] # Terminate game
                del game_tokens[game] # Terminate game
                if len(game_boards) == 0:
                    return tournament_results

            # Select end_token based on scores and topk
            selected_end_tokens = {game:self.selector(game_end_tokens[game], game_end_scores[game], self.p, self.k) for game in game_boards}
            # Find end_board corresponding to selected end_tokens
            selected_end_boards = {game:options[game_tokens[game]][end_token] for game,end_token in selected_end_tokens.items()}

            # Add this move to the game_record
            for game in game_boards:
                tournament_results[game][color_toplay]['moves'].append((selected_end_tokens[game], selected_end_boards[game]))

            # Swap to opponent's perspective and conjugate board ready for opponent play
            color_toplay = 'white' if color_toplay == 'black' else 'black' # Swap player turns
            game_boards = {game: conjugate_board(selected_end_boards[game]) for game in game_boards}
            game_tokens = {game: board_token(board, f'{color_toplay}S_') for game,board in game_boards.items()}

    def selector(self, tokens, scores, p=0.3, k=3):
        '''
        This is an elegant idea. Squash the choice distribution to have a target (lower) entropy.
        Selects a token, based on log2(p * len(k)) degrees of freedom.
        '''
        if len(set(scores)) == 1: # If there is no variance in the scores, then just chose randomly.
            return choices(tokens)[0]
        else:
            # Otherwise target entropy is either proportion p * max_possible_entropy (for small option sets) or as-if k-degree of freedom distribution (for scores >> k)
            target_entropy = min(p * np.log2(len(scores)), np.log2(k))
            t = entropy_temperature(scores, target_entropy)
            dist = softmax_temp(scores, temp=t)
            return choices(tokens, cum_weights=list(accumulate(dist)))[0]
        

def summarise_sim_tournament(tournament_results):
    # tournament_results = {game: {'white': {'points': 0, 'moves':[(token, board), ...]}, 'black': {...}}, ...}
    sim_summary = dict()
    for game in tournament_results:
        for color in tournament_results[game]:
            points = tournament_results[game][color]['points']
            for token,_ in tournament_results[game][color]['moves']:
                if token in sim_summary:
                    sim_summary[token]['visits'] += 1
                    sim_summary[token]['points'] += points
                else:
                    sim_summary[token] = {'visits':1, 'points':points}
    return sim_summary


def main(
        tournament_id, agents_spec=None, num_games=None, starting_state=None, max_moves=float('inf'), save=False, result_dest=None
    ):
    '''
    main function should build the model and call the play_tournament_master function on it
    tournament_id is a dummy variable that eats the dummy input from the pool.map function
    '''
    # agent scope: model, num_simgames, max_simmoves, C, p, k
    # agents_spec = [{'type':'dummy', 'kwargs':kwargs, 'num_simgames':#, 'max_simmoves':#, 'C':#, 'p':#, 'k':#}, {...}]

    agents = []
    for spec in agents_spec:
        # init model
        if spec['type'] is None:
            model = None
        elif spec['type'] == 'pieces':
            model = PiecesModel(**spec['kwargs'])
        elif spec['type'] == 'chessai':
            model = ChessAI(**spec['kwargs'])
        elif spec['type'] == 'transformer':
            model = TransformerModel(**spec['kwargs'])
        else:
            assert 1 == 0, f'Agent "type" must be from ["dummy", "chessai" or "transformer"] but {spec["type"]} was passed.'

        agent = Agent(model, **{k:v for k,v in spec.items() if k not in ['type','kwargs']})
        agents.append(agent)

    # Play the dang tournament! model_scores = [float, float]; game_tree, board_scores results are not used.
    kwargs = {'agents':agents, 'num_games':num_games, 'starting_state':starting_state, 'max_moves':max_moves}
    tournament_results = play_tournament(**kwargs)

    # Save tournament_results for retrieval later in training
    if save:
        with open(os.path.join(result_dest, f'tmnt_{uuid.uuid4().hex}.pkl'), 'wb') as pkl:
            pickle.dump(tournament_results, pkl)

if __name__ == "__main__":
    '''
    Notebook script should be able to access namespace including the following:
    num_workers          # Required: number of python processes to spawn.
    num_tournaments      # Required: number of tournaments to play - one worker per tournament
    agents_spec          # Required: specification for each agent including model and selector params
    num_games            # Required: number of games to play in sequence each tournament
    starting_state       # Can be None: starting state for tournaments in the form of (board, color_toplay)
    max_moves            # Required: max number of master-level game moves per game
    save                 # Required: boolean - whether to save the self-play results or not
    result_dest          # Optional based on "save": destination directory for tournament results to be saved
    '''
    
    # Assume that the following variables are defined in the namespace of the controlling notebook:
    kwargs = {
        'agents_spec': agents_spec, 'num_games': num_games, 'starting_state': starting_state, 'max_moves': max_moves, 'save': save, 'result_dest': result_dest
    }

    # agents_spec=None, num_games=None, starting_state=None, max_moves=float('inf'), save=False, result_dest=None
    
    # Bind user input variables to the main function
    main_ = partial(main, **kwargs)

    # Call the script that will play num_games in parallel
    with Pool(num_workers) as pool:
        _ = pool.map(main_, range(num_tournaments))