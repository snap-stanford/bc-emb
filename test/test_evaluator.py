from recsys.dataset import DynRecDataset
import torch

def test_recallevaluator():
    from recsys.evaluate import RecallEvaluator
    from recsys.utils import split_dynrecdataset

    dataset = DynRecDataset('Amazon-Video_Games')
    edge_index_useritem_list, num_users_list, num_items_list\
        = split_dynrecdataset(dataset, [0.8, 0.9, 1.0], exclude_new_users = False)
    
    edge_index_useritem_dict, num_users_dict, num_items_dict = {}, {}, {}
    split_list = ['train', 'val', 'test']
    for i, split in enumerate(split_list):
        edge_index_useritem_dict[split] = edge_index_useritem_list[i]
        num_users_dict[split] = num_users_list[i]
        num_items_dict[split] = num_items_list[i]

    evaluator = RecallEvaluator(edge_index_useritem_dict, num_users_dict, num_items_dict)

    print(num_users_dict)
    print(num_items_dict)
    print(edge_index_useritem_dict)
    print([(key, value.shape) for key, value in edge_index_useritem_dict.items()])

    # test train mode
    user_idx = torch.arange(100)
    rating = torch.rand(100, evaluator.num_items_dict['train'])
    eval_users_idx, recall_dict = evaluator.eval(rating, user_idx, split = 'train')
    print(eval_users_idx, recall_dict)

    # test val mode
    user_idx = torch.arange(100)
    rating = torch.rand(100, evaluator.num_items_dict['val'])
    eval_users_idx, recall_dict = evaluator.eval(rating, user_idx, split = 'val')
    print(eval_users_idx, recall_dict)

    # test test mode
    user_idx = torch.arange(100)
    rating = torch.rand(100, evaluator.num_items_dict['test'])
    eval_users_idx, recall_dict = evaluator.eval(rating, user_idx, split = 'test')
    print(eval_users_idx, recall_dict)

def test_recallevaluator_notest():
    # when there is no test data.
    from recsys.evaluate import RecallEvaluator
    from recsys.utils import split_dynrecdataset

    dataset = DynRecDataset('Amazon-Video_Games')
    edge_index_useritem_list, num_users_list, num_items_list\
        = split_dynrecdataset(dataset, [0.8, 1.0], exclude_new_users = True)
    
    edge_index_useritem_dict, num_users_dict, num_items_dict = {}, {}, {}
    split_list = ['train', 'val']
    for i, split in enumerate(split_list):
        edge_index_useritem_dict[split] = edge_index_useritem_list[i]
        num_users_dict[split] = num_users_list[i]
        num_items_dict[split] = num_items_list[i]

    evaluator = RecallEvaluator(edge_index_useritem_dict, num_users_dict, num_items_dict)

    print(num_users_dict)
    print(num_items_dict)
    print(edge_index_useritem_dict)
    print([(key, value.shape) for key, value in edge_index_useritem_dict.items()])

    # test train mode
    user_idx = torch.arange(100)
    rating = torch.rand(100, evaluator.num_items_dict['train'])
    eval_users_idx, recall_dict = evaluator.eval(rating, user_idx, split = 'train')
    print(eval_users_idx, recall_dict)

    # test val mode
    user_idx = torch.arange(100)
    rating = torch.rand(100, evaluator.num_items_dict['val'])
    eval_users_idx, recall_dict = evaluator.eval(rating, user_idx, split = 'val')
    print(eval_users_idx, recall_dict)

def test_downstream_evaluator():
    from recsys.evaluate import UnintendedTaskEvaluator

    evaluator = UnintendedTaskEvaluator(metric = 'binacc')
    y_pred = torch.Tensor([0.2, -0.3, 4, 0.9, -0.9, 0.2])
    y_true = torch.LongTensor([1, 0, 1, 1, 0, 1])
    acc = evaluator.eval(y_pred, y_true)
    print(acc)

    evaluator = UnintendedTaskEvaluator(metric = 'rocauc')
    rocauc = evaluator.eval(y_pred, y_true)
    print(rocauc)

    evaluator = UnintendedTaskEvaluator(metric = 'rmse')
    y_pred = torch.randn(10)
    y_true = torch.rand(10)
    rmse = evaluator.eval(y_pred, y_true)
    print(rmse)

    evaluator = UnintendedTaskEvaluator(metric = 'multiacc')
    y_true = torch.randint(0, 3, size = (10,))
    y_pred = torch.rand(10, 3)
    y_pred[torch.arange(len(y_true)), y_true] = 3
    acc = evaluator.eval(y_pred, y_true)
    print(acc)


if __name__ == '__main__':
    # test_recallevaluator()
    # test_recallevaluator_notest()
    test_downstream_evaluator()