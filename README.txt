Key files:
 - chess_primitives.py - calculations necessary for the game of chess.
 - chess_model.py - home for the model classes experimenting with.
 - chess_selfplay.py - functions for executing a chess tournament using multiprocessing
 - chess_training.py - functions for compiling datasets from self-play and training a model
 - ContinuousTraining.ipynb - main loop for self-play / training / self-play / training ...
 - ProgressInspection.ipynb - functions for peeking in at the progress of self-play and training steps each round.

data -> output -> ...

--round_0
   |--checkmates.pkl
   |--model.pt
   |--self_play
   |   |--tnmt_XXXX.pkl
   |   |--tnmt_YYYY.pkl
   |   ...
--round_1
   | ...

