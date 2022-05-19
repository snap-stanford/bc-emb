sliced_time_list = [0.5, 0.6, 0.7, 0.8, 0.9, 1]

def get_pinsage_hyperparam_list(dataset_name: str):
    if dataset_name in ['Amazon-Musical_Instruments', 'Amazon-Video_Games']:
        return [
            {'num_layers': 2, 'emb_dim': 256}, 
            {'num_layers': 2, 'emb_dim': 320}, 
            {'num_layers': 3, 'emb_dim': 384},
            {'num_layers': 3, 'emb_dim': 448},
            {'num_layers': 3, 'emb_dim': 512},
        ]
    elif dataset_name in ['Amazon-Grocery_and_Gourmet_Food']:
        return [
            {'num_layers': 1, 'emb_dim': 256}, 
            {'num_layers': 1, 'emb_dim': 320}, 
            {'num_layers': 2, 'emb_dim': 384},
            {'num_layers': 2, 'emb_dim': 384},
            {'num_layers': 2, 'emb_dim': 384},
        ]
    else:
        raise ValueError(f'Unknown dataset name called {dataset_name}')
