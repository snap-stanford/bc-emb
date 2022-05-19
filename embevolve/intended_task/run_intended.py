device = 0

dataset_list = [
    'Amazon-Musical_Instruments',
    'Amazon-Video_Games',
    'Amazon-Grocery_and_Gourmet_Food',
]

lam=16

strategy_list = [
    'fix-m0',
    'finetune-m0',
    'nobc',
    'posthoc-linear-sloss',
    'posthoc-linear-mloss',
    f'joint-linear-sloss-lam{lam}',
    f'joint-notrans-sloss-lam{lam}',
    f'joint-linear-mloss-lam{lam}',
]

for dataset in dataset_list:
    for strategy in strategy_list:
        config=f'code-release/{dataset}/{strategy}'
        print(f'python generate_embeddings.py --strategy {strategy} --dataset {dataset} --log_dir runs/{config} --checkpoint_path checkpoint/{config} --device {device}')



