# packages
import os, torch
from chess_model import TransformerModel, ChessDataset, TanhLoss, train
from torch.utils.data import random_split, DataLoader
from functools import partial
from multiprocessing import Pool

def main(pid, root_dir, current_training_round_dir, model_kwargs, device):
    latest_model_path = os.path.join(root_dir, current_training_round_dir, 'model.pt')
    if not (os.path.exists(latest_model_path) and os.path.isfile(latest_model_path)):

        # No model saved here yet. Create and train a new model based on the previous k rounds of self-play data.
        # We currently have 591 tournaments saved in baseline and 191 in round1. We could use k = 10 and go from there?
        model_kwargs['load_path'] = None # Brand new model
        model = TransformerModel(**model_kwargs)
        optimizer = torch.optim.Adam(model.parameters(), lr=0, weight_decay=0)
        loss_fn = TanhLoss()
        dataset = ChessDataset(root_dir=root_dir, look_back=10, device=device)
        train_set, test_set = random_split(dataset, [int(len(dataset)*0.8), len(dataset) - int(len(dataset)*0.8)])
        train_loader = DataLoader(train_set, batch_size=1000, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_set, batch_size=1000, shuffle=True, num_workers=0)
        print(f'Training on {len(train_set):,.0f} examples in {len(train_loader):,.0f} batches.')

        # Train on the data
        model_dest = os.path.join(root_dir, current_training_round_dir, 'model.pt')
        model = train(model, loss_fn, optimizer, train_loader, test_loader, warmup_passes=4, max_lr=1e-4, save_dir=model_dest, stopping=10)
        
        # Cut ties with now orphan variables
        del model, optimizer, loss_fn, dataset, train_set, test_set, train_loader, test_loader

    # Clean up CUDA memory
    torch.cuda.empty_cache()

if __name__ == "__main__":
    '''
    Notebook script should be able to access self_play_args dictionary object specifying the following:
    root_dir, current_training_round_dir, model_kwargs, device
    '''

    # Bind user input variables to the main function
    main_ = partial(main, **modeltraining_args)

    # Call the script that will play num_games in parallel
    with Pool(1) as pool:
        _ = pool.map(main_, range(1))

