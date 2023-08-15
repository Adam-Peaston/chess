# packages
import os, torch, pickle
import numpy as np
from chess_model import TransformerModel, ChessDataset, TanhLoss
from torch.utils.data import DataLoader
from functools import partial
from multiprocessing import Pool

def _compile_dataset_(root_dir, sub_dirs, discount=0.5):
    factors = [discount**i for i in range(len(sub_dirs))][::-1]
    data = {}
    for sub_dir,factor in zip(sub_dirs, factors):
        self_play_path = os.path.join(root_dir, sub_dir, 'self_play')
        tournamentfiles = [f for f in os.listdir(self_play_path) if f.startswith('tmnt_') and f.endswith('.pkl')]
        for file in tournamentfiles:
            with open(os.path.join(self_play_path, file), 'rb') as pkl:
                tourn = pickle.load(pkl)
            for i, pair in tourn.items():
                for order,game in pair.items():
                    for color in game:
                        points = game[color]['points']
                        for token,board in game[color]['moves']:
                            if token in data:
                                data[token]['visits'] += factor
                                data[token]['points'] += points * factor
                            else:
                                data[token] = {'board': board, 'visits': factor, 'points': points * factor}
    return data

def compile_datasets(root_dir, look_back=20, discount=0.7):
    # Catalogue training rounds to source dataset from
    training_round_dirs = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir,d))], key=lambda d: int(d.split('_')[-1]))
    # Last two self-play directories to be used for training and test set respectively
    train_dirs = training_round_dirs[-look_back:-1]
    test_dir = training_round_dirs[-1]
    # Compile training positions
    train_data = _compile_dataset_(root_dir, train_dirs, discount)
    test_data = _compile_dataset_(root_dir, [test_dir], discount)
    
    # augment with checkmates from all previous training rounds, except test round
    for round_dir in training_round_dirs[:-1]:
        checkmates_file = os.path.join(root_dir, round_dir, 'checkmates.pkl')
        with open(checkmates_file, 'rb') as pkl:
            checkmates = pickle.load(pkl)
        for token,board,points in checkmates:
            if token in train_data:
                train_data[token]['visits'] = 1
                train_data[token]['points'] = points
            else:
                train_data[token] = {'board':board, 'visits':1, 'points':points}

    return train_data, test_data

def train(model, loss_fn, optimizer, train_dataloader, test_dataloader, warmup_passes, max_lr, save_dir, slope_threshold=0, stop_after=10):

    # A little more room to keep training until we're sure there's no more improvement to be found. Fit a line to the latter half of recorded test losses.
    # If the slope of that line is greater than some threshold (0 indicates no average improvement), then increment the stopping counter, ultimately halting. 

    warm_steps = len(train_dataloader) * warmup_passes
    train_step = 0
    best_test_loss = float('inf')
    test_losses = []
    stopping_count = 0
    epoch = 0
    slope = 0
    train_report = []

    while stopping_count < stop_after:
        
        epoch += 1
        loss_sum = 0
        examples_seen = 0

        for data in train_dataloader:
            optimizer.zero_grad()
            X, y = data
            z = model(X)
            loss = loss_fn((z, y))
            loss_sum += loss.item()
            examples_seen += X.shape[0]
            loss.backward()
            optimizer.step()
            # learning rate warm-up
            train_step += 1
            for g in optimizer.param_groups:
                g['lr'] = min(max_lr, max_lr * train_step / warm_steps)
        mean_train_loss = loss_sum / examples_seen

        with torch.no_grad():
            loss_sum = 0
            examples_seen = 0
            for data in test_dataloader:
                X, y = data
                z = model(X)
                loss = loss_fn((z, y))
                loss_sum += loss.item()
                examples_seen += X.shape[0]
            mean_test_loss = loss_sum / examples_seen
            test_losses.append(mean_test_loss)

            if mean_test_loss < best_test_loss:
                torch.save(model.state_dict(), os.path.join(save_dir, 'model.pt'))
                best_test_loss = mean_test_loss

            if epoch >= 4: # only a valid measure after 4 epochs
                # Fit a regression line to the second half of the recorded test_losses
                X = np.linspace(0, 1, len(test_losses)//2)
                y = np.array(test_losses[-len(X):])
                y = (y - y.min()) / (y.max() - y.min()) # Squish to range [0, 1]
                slope = ((X-X.mean())*(y-y.mean())).sum()/(((X-X.mean())**2).sum()) # Slope of best fit line
                if slope > slope_threshold:
                    stopping_count += 1
                else:
                    stopping_count = 0

        print(f'Epoch: {epoch}, train loss: {mean_train_loss:,.5f}, test loss: {mean_test_loss:,.5f}, slope: {slope:,.3f}, stopping count: {stopping_count}')
        train_report.append([epoch, mean_train_loss, mean_test_loss, slope, stopping_count])
        with open(os.path.join(save_dir, 'report.pkl'), 'wb') as pkl:
            pickle.dump(train_report, pkl)

    return model


def main(pid, mode, root_dir, previous_training_round_dir, current_training_round_dir, model_kwargs, device):

    previous_model_path = os.path.join(root_dir, previous_training_round_dir, 'model.pt')
    if mode == 'continuous' and os.path.exists(previous_model_path) and os.path.isfile(previous_model_path):
        print('Training from model saved last round.')
        model_kwargs['load_path'] = previous_model_path # Load previous generation model
    else:
        print('Training brand new model.')
        model_kwargs['load_path'] = None # Brand new model
    
    model = TransformerModel(**model_kwargs)
    optimizer = torch.optim.Adam(model.parameters(), lr=0, weight_decay=0)
    loss_fn = TanhLoss()

    train_data, test_data = compile_datasets(root_dir, look_back=20, discount=0.7)
    train_set = ChessDataset(train_data, device=device)
    test_set = ChessDataset(test_data, device=device)

    train_loader = DataLoader(train_set, batch_size=1000, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=1000, shuffle=True, num_workers=0)
    print(f'Training on {len(train_set):,.0f} examples in {len(train_loader):,.0f} batches.')

    # Train on the data
    model_dest = os.path.join(root_dir, current_training_round_dir)
    model = train(model, loss_fn, optimizer, train_loader, test_loader, warmup_passes=4, max_lr=2e-4, save_dir=model_dest, slope_threshold=0, stop_after=20)
    
    # Clean up CUDA memory
    del model, optimizer, loss_fn, train_set, test_set, train_loader, test_loader
    torch.cuda.empty_cache()


if __name__ == "__main__":
    '''
    Notebook script should be able to access self_play_args dictionary object specifying the following:
    root_dir, current_training_round_dir, model_kwargs, device
    '''

    # Bind user input variables to the main function
    main_ = partial(main, **modeltraining_args)

    # Only need to spin up one process, just using this to completely tear down memory allocation at end of training.
    with Pool(1) as pool:
        _ = pool.map(main_, range(1))

