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

prediction_task_list = [
    'binary-edge-rating',
    'user-activity',
    'user-positive-activity',
    'binary-item-rating-avg',
    'binary-item-rating-std',
]


for dataset in dataset_list:
    print()
    for prediction_task in prediction_task_list:
        print()
        dirname = f'code-release/{dataset}/{prediction_task}'
        model_path = f'checkpoint/{dirname}/model_list'

        # Tune hyper-parameters and save models for the unintended task
        print(f'python train_model.py --dataset {dataset} --prediction_task {prediction_task} --model_path {model_path} --device {device}')

        # Evaluate the saved models on different embgen strategies
        for strategy in strategy_list:
            result_path = f'result/{dirname}/{strategy}'
            print(f'python eval_model.py --embgen_strategy {strategy} --dataset {dataset} --prediction_task {prediction_task} --model_path {model_path} --result_path {result_path} --device {device}')



