# packages
import os, torch
from chess_model import TransformerModel, ChessDataset, TanhLoss, compile_dataset, train
from torch.utils.data import random_split, DataLoader
from functools import partial
from multiprocessing import Pool

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

    train_data, test_data = compile_dataset(root_dir, look_back=10, tts=0.8)
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

    # Call the script that will play num_games in parallel
    with Pool(1) as pool:
        _ = pool.map(main_, range(1))

