# chess
My goal here is to create an AI that learns to play chess from scratch, or "tabula rasa". This project inculdes development of necessary chess primitives like boards, pieces, allowed moves, games, etc. as well as the core AI which will be a Transformer model, and necessary data generation and training loops.

Let's see how we go!

## Background
I have always found the concept of machine intelligence fascinating. Furthermore, when machines are able to craft an action policy based on nothing but their own experiences, that seems to be to be something more than simply holding up a mirror to humanity's domain experts or humanity as a whole.

I am basing this project on the work of Silver et. al. at DeepMind who demonstrated a general purpose RL algorithm, AlphaZero, that was capable of achieving SOTA performance on the games of Chess, Shogi, and Go: https://arxiv.org/pdf/1712.01815.pdf.

Some variations on their work that I am interested to explore in this project:
1. The AlphaZero algorithm, while not specific to any particular model architecture, was demonstrated using a CNN architecture. I am keen to see how well a Transformer architecture will take to this task.
2. The AlphaZero RL approach involved generating two outputs from a given board state; (1) a vector of move probabilities p = P(a|s), and (2) a scalar representing the likelihood of winning from this board state v = E[z|s]. I have a couple of challenges for this. The vector of move probabilities needs to have space for all possible moves, regardless of board state, which is then presumably filtered through some kind of mask to permit the model to make only legal moves. This vector must be extremely long, and does not absolve the programmer of developing the mask anyway. I hypothesise it would be more efficient to constrain the model's attention at the outset to only legal moves. The scalar output as described does not seem to be contingent on the actually selected move, but is obviously jointly contingent on the output vector of move probabilities. Therefore the scalar is not (really) E[z|s] but rather E[z|s,p]. The best that p can add to this estimator is some indication of which move will actually be made. As a result, the AlphaZero formulation seems indirect, and potentially leaves room for efficiency improvement. Therefore, in this work I am experimenting with simplifying the role of the model to only evaluating board positions, and evaluating end-of-move positions rather than starting positions. The task of the model is simply to estimate, for each possible move, the likelihood of winning the game if that move is made.

While the Attention mechanism is a relatively minor incidental aspect of the overall project, if successful I would title the final paper on this work "Evaluation Is All You Need".

## Lessons learnt
1. Achieving checkmates is important for generating training data. This seems obvious in retrospect, but position evaluations provide greater learning signal when annotated with either a win or a loss outcome - draw outcomes have the opposite effect. This explains the wisdom of constructing agents that can be parameterised with greater and lesser relative strength to increase the likelihood of a game ending in checkmate.
