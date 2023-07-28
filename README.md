# chess
My goal here is to create an AI that learns to play chess from scratch, or "tabula rasa". This project inculdes development of necessary chess primitives like boards, pieces, allowed moves, games, etc. as well as the core AI which will be a Transformer model, and necessary data generating and training loops.

Let's see how we go! All suggestions and collaboration proposals are welcome!

## Background
I have always found the concept of machine intelligence fascinating. Furthermore, when machines are able to craft an action policy based on nothing but their own experiences, that seems to be to be something more than simply holding up a mirror to humanity's domain experts or humanity as a whole.

I am basing this project on the work of Silver et. al. at DeepMind who demonstrated a general purpose RL algorithm, AlphaZero, that was capable of achieving SOTA performance on the games of Chess, Shogi, and Go: https://arxiv.org/pdf/1712.01815.pdf.

Some variations on their work that I am interested to explore in this project:
1. The AlphaZero algorithm, while not specific to any particular model architecture, was demonstrated using a CNN architecture. I am keen to see how well a Transformer architecture will take to this task.
2. The AlphaZero RL approach involved generating two outputs from a given board state;
 - a vector of move probabilities $\ p = P(a|s)$, and
 - a scalar $\ v$ representing the likelihood of winning ($\ outcome = z$) from this board state $\ v = E[z|s]$.
I have a couple of challenges for this:
 - The vector of move probabilities needs to have space for all possible moves, regardless of board state, which is then presumably filtered through a mask to permit the agent to make only legal moves. This vector must be extremely long, and does not absolve the programmer of developing the mask for each position anyway. I hypothesise it would be more efficient to constrain the model's attention at each position to only legal moves from that position.
 - The scalar output as described does not seem to be contingent on the actually selected move, but is instead jointly contingent on the output vector of move probabilities. Therefore the scalar $\ v$ I argue is not (really) $\ E[z|s]$ but rather $\ E[z|s,p]$. The best that $\ p$ can add to this estimator is some indication of which move will _actually_ be made. As a result, the AlphaZero formulation seems indirect, and potentially leaves room for efficiency improvement. Therefore, in this work I am experimenting with simplifying the role of the model to only evaluating board positions, and evaluating end-of-move positions rather than starting positions. The task of the model is then simply to estimate, for each possible move, the likelihood of winning the game from that move-end state.
4. I would like my model to develop with training over a number of days at most, and have not yet decided to invest in cloud / cluster infrastructure. This is as much for fun (to see what can be achieved on a small laptop) as it is for economic reasons. I have not yet conceived of a way to run the AlphaZero search algorithm (as opposed to MCTS) in a way that parallelises nicely, as in the AlphaZero algorithm each move expansion decision necessarily (it seems to me) depends on the series of previous expansions. Therefore I am reverting back (from the AlphaZero algorithm) to stochastic move expansions that more closely resembles MCTS as this can be more easily parallised and augmented with multiprocessing.
5. A blog I read today (25 July 2023) by Evan Miller hypothesised that certain arguably undesireable phenomena in the learned weights of transformer models has to do with the fact that the softmax function applied within the scaled dot product attention mechanism _requires_ each attention head to output an activation vector which sums to 1, even if the specialisation of that head has little to say about the token being processed. Instead Evan suggests a modification of the softmax function, whimsically called the _ghostmax_ function which would enable attention heads to output activation vectors which sum to less than 1. This sounds worth an experiment to me, and I am interested to see if this modification could obviate the need for learning rate warm-up when training transformer models, as the attenion heads could first learn to be quiet before attempting to add any useful information. I also like the title of this blog post "Attention Is Off By One" https://www.evanmiller.org/attention-is-off-by-one.html

While the Attention mechanism is a relatively minor incidental aspect of the overall project, if successful I would title the final paper on this work "Evaluation Is All You Need".

## Methodology
1. Develop a baseline dataset of games played through self-play of a relatively weak heuristic model. The heuristic model is necessary to ensure a sufficient proportion of self-play games end in a win or loss outcome that has a good learning signal, within a reasonable number of moves (max set at 200 moves).
2. Train the first Transformer model on the baseline dataset, using early stopping for regularisation based on stagnation of model performance on a hold-out/test set.
3. Self-play with two agents powered by the same Transformer model - one with deep and broad look-ahead, and one without any lookahead really at all - to generate the round-1 dataset.
4. Either re-train a new Transformer model on the round1 dataset, or fine-tune the first model on the round1 dataset. Here some experimentation will be required. It will be interesting to see if this model/application exhibits "catastrophic forgetting" during fine-tuning, which would be a good reason to train a new model each round from scratch. Another potentially good reason to start with a new model each round would be that we could increase or adapt the model hyper-parameters each round to find new architectures that are better able to model the new round of self-play data. 

## Lessons learnt
1. Confidence is key. Simply applying softmax to raw move option scores produces a distribution that, if sampled from with no further adjustment, produces too great a variety of moves. Essentially the model is typically not confident enough (yet) to back itself, and so moves selected from within this uncertainty resemble purely random move selection. To address this, I have developed a technique for adjusting the move probability distirbution with a temperature parameter to achieve a target Shannon entropy of log2(k) for distributions of size >> k, or p * log2(size of distribution) whichever is the minimum, for some float parameter k and some percentage parameter p... (update: Actually experiment shows that the k parameter might be superfluous, at least during self-play. In fact it may even be better in general to allow the model to sample from a higher-entropy distribution where there are more move options available. This is especially the case during self-play where we want the agent to explore and return game outcomes from a wider set of possible positions. Durinh competition play on the other hand we would presumably just run an argmax agent, or set p and/or k to some small values.). Setting k = 3 means the move will almost certainly be chosen from among the top 3 (ish) options. Where there are fewer than 3 options available, the parameter p governs and the distribution will be adjusted to have p% of the maximum possible entropy for that size of distribution. This technique boosts the move selection confidence in a way that respects relative magnitude of raw move scores, and focusses the move selection on the top k (ish) scores, while in prininciple not eliminating the possibility of making any of the potential moves from a given position.
2. Achieving checkmates is important for generating training data. This seems obvious in retrospect, but position evaluations provide greater learning signal when annotated with either a win or a loss outcome - draw outcomes have the opposite effect. This explains the wisdom of constructing agents that can be parameterised with greater and lesser relative strength to increase the likelihood of a game ending in checkmate.
3. Counterbalancing point 3 is the need to encourage diversity of move selection during self-play to generate training data. This is important in order to train a model that generalises to positions that are not central to the set of likely positions given the model.
