for dataset in ['ml-25m']: # ['Amazon-Musical_Instruments', 'Amazon-Video_Games']: #, ]:
    print(f'sh dec24.sh {dataset} train-once 1;')
    print(f'sh dec24.sh {dataset} train-from-scratch-vanilla 3;')
    print(f'sh dec24.sh {dataset} train-from-scratch-align-linear-l2 4;')

print()

# gpu_list = [1,3,4,6]
# for i, lam in enumerate([1,4,8,16]): #, 8]):
#     for dataset in ['ml-25m']: # 'Amazon-Musical_Instruments', 'Amazon-Video_Games']: #]:
#         print(f'sh dec24.sh {dataset} train-from-scratch-penalize-l2-{lam} {gpu_list[i]};')
#         print(f'sh dec24.sh {dataset} train-from-scratch-penalize-linear-l2-{lam} {gpu_list[i]};')
#         # print(f'sh dec24.sh {dataset} train-from-scratch-finetune-penalize-linear-l2-{lam} {gpu_list[i]};')
#         print(f'sh dec24.sh {dataset} train-from-scratch-timeaug-penalize-l2-{lam} {gpu_list[i]};')
#         print(f'sh dec24.sh {dataset} train-from-scratch-timeaug-penalize-linear-l2-{lam} {gpu_list[i]};')
#         # print(f'sh dec24.sh {dataset} train-from-scratch-amplifypenalize-linear-l2-{lam} {gpu_list[i]};')
#     print()