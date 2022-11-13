from copy import deepcopy

from torch.utils.data import DataLoader, ConcatDataset
from datasets.utils import FullDatasetBase

__all__ = ["DataProvider", "DatasetFactory"]


class DataProvider:
    train_dl: DataLoader
    val_dl: DataLoader
    test_dl: DataLoader

    def __init__(self, params: dict):
        params = deepcopy(params)
        self.factory = DatasetFactory()
        self.dataset, self.dataset_params, train, val, test = self.factory.build_dataset(params)

        if 'workers' in params:
            workers = params['workers']
            params.pop('workers')
        else:
            workers = 4
        if 'adv_dataset' in self.dataset_params and self.dataset_params['adv_dataset']:
            print("Since using adv dataset which needs cuda, so using dataloader in main process with worker=0.")
            workers = 0

        train_bz = val_bz = test_bz = params['batch_size']
        params.pop('batch_size')
        if 'train_bz' in params:
            train_bz = params['train_bz']
        if 'test_bz' in params:
            val_bz = test_bz = params['test_bz']

        self.train_dl = self._create_dataloader(train, shuffle=True, workers=workers, batch_size=train_bz, **params)
        if 'repeat' in params:
            params.pop('repeat')
        self.val_dl = self._create_dataloader(val, shuffle=False, workers=workers, batch_size=val_bz, **params)
        self.test_dl = self._create_dataloader(test, shuffle=False, workers=workers, batch_size=test_bz, **params)

    @staticmethod
    def _create_dataloader(base_dataset, workers=4, batch_size=256, drop_last=False, shuffle=False, repeat=1,
                           collate_fn=None, **kwargs):
        if repeat > 1:
            base_dataset = ConcatDataset([base_dataset] * repeat)
        loader = DataLoader(base_dataset, batch_size=batch_size, num_workers=workers, shuffle=shuffle,
                            drop_last=drop_last, pin_memory=True, collate_fn=collate_fn)
        print(f"dataset len: {len(base_dataset)}")
        return loader


class DatasetFactory:
    from datasets.nerf.nerf_dataset import NeRFFullDataset

    dataset_params: dict
    all_datasets = [NeRFFullDataset]

    @staticmethod
    def build_dataset(dataset_params):
        dataset_name = dataset_params['name']
        assert isinstance(dataset_name, str)

        if dataset_name == "concat":
            assert isinstance(dataset_params['sub_datasets'], list) or isinstance(dataset_params['sub_datasets'], tuple)
            from datasets.decorators import ConcatFullDataset
            dataset = ConcatFullDataset(dataset_params)
            return dataset, dataset_params, dataset.train, dataset.val, dataset.test
        else:
            dataset_type, dataset_params = DatasetFactory.analyze_name(dataset_name, dataset_params)
            # print("dataset_params:", dataset_params)  # closed due to large output from nerf
            dataset = dataset_type(**dataset_params)
            train, val, test = DatasetFactory.gen_datasets(dataset, dataset_params)
            return dataset, dataset_params, train, val, test

    @staticmethod
    def analyze_name(name: str, params, type_only=False):
        import re

        if "dataset_mapping" not in params:
            params["dataset_mapping"] = (0, 1, 2)
        if "dataset_transforms" not in params:  # 0-> train transforms 1-> test transforms
            params["dataset_transforms"] = (0, 1, 1)

        dataset_type = None
        while True:
            for d in DatasetFactory.all_datasets:
                if d.is_dataset_name(name):
                    dataset_type = d
                    break
            if name.endswith("_partial"):
                name = name[:-8]
                params["partial_train"] = True
            elif name.endswith("_test"):
                name = name[:-5]
                params["dataset_mapping"] = (2, 2, 2)
            elif name.endswith("_train"):
                name = name[:-6]
                params["dataset_mapping"] = (0, 0, 0)
            elif name.endswith("_val"):
                name = name[:-4]
                params["dataset_mapping"] = (1, 1, 1)
            elif name.endswith("_swap"):
                name = name[:-5]
                a, b, c = params["dataset_mapping"]
                params["dataset_mapping"] = (b, c, a)
            elif name.endswith("_noaug"):
                name = name[:-6]
                params["dataset_transforms"] = (1, 1, 1)
            elif name.endswith("_allaug"):
                name = name[:-7]
                params["dataset_transforms"] = (0, 0, 0)
            elif name.endswith("_ordered"):
                name = name[:-8]
                params['order_all'] = True
            elif re.match("(Order|order|o|O)(Cifar|cifar|CIFAR)([-_])*100$", name):
                name = "cifar100_ordered"
            elif re.match("(Order|order|o|O)(Cifar|cifar|CIFAR)([-_])*10$", name):
                name = "cifar10_ordered"
            elif name.endswith("_adv"):
                name = name[:-4]
                params["adv_dataset"] = True
            elif name.endswith('_rand'):
                name = name[:-5]
                params["rand_dataset"] = True
            else:
                break

        if dataset_type is None:
            raise NotImplementedError("Dataset Not Implemented")

        if type_only:
            return dataset_type
        else:
            return dataset_type, params

    @staticmethod
    def gen_datasets(dataset: FullDatasetBase, params):
        trans = [dataset.gen_train_transforms(), dataset.gen_test_transforms()]
        data_gens: list = [dataset.gen_train_datasets, dataset.gen_val_datasets, dataset.gen_test_datasets]

        if 'order_all' in params and params['order_all']:
            from datasets.decorators import OrderDataset

            def apply_order(data_gen):
                def order(*input, **kwargs):
                    data = data_gen(*input, **kwargs)
                    from torchvision.datasets.cifar import CIFAR10
                    from torchvision.datasets.cifar import CIFAR100
                    from torchvision.datasets import ImageFolder
                    if isinstance(data, (CIFAR10, CIFAR100, ImageFolder)):
                        return OrderDataset(data)
                    else:
                        raise NotImplementedError("Ordered dataset for this dataset is not implemented.")
                return order

            data_gens = [apply_order(d) for d in data_gens]

        if 'partial_train' in params and params['partial_train']:
            from datasets.decorators import PartialDataset
            assert 'total' in params and 'selected' in params

            def apply_partial(data_gen):
                def partial(*input, **kwargs):
                    train_data = data_gen(*input, **kwargs)
                    return PartialDataset(train_data, total=params['total'], selected=params['selected'])
                return partial

            data_gens[0] = apply_partial(data_gens[0])

        if 'adv_dataset' in params and params['adv_dataset']:
            from datasets.decorators import AdvDataset
            if not isinstance(params["adv_path"], str):
                assert KeyError('params["adv_path"] must be a str that indicates the path of model for adv')
            if not isinstance(params["attack_type"], str):
                assert KeyError('params["attack_type"] must be a str that indicates type of attack')

            def apply_adv(data_gen):
                def adv(*input, **kwargs):
                    data = data_gen(*input, **kwargs)
                    return AdvDataset(data, dataset.mean, dataset.std, params["adv_path"], params["attack_type"])
                return adv

            data_gens = [apply_adv(d) for d in data_gens]

        if 'rand_dataset' in params and params['rand_dataset']:
            from datasets.decorators import RandDataset

            def apply_rand(data_gen):
                def rand(*input, **kwargs):
                    data = data_gen(*input, **kwargs)
                    return RandDataset(data, params['alpha'])
                return rand

            data_gens = [apply_rand(d) for d in data_gens]

        train, val, test = [data_gens[params['dataset_mapping'][i]](*trans[params['dataset_transforms'][i]]) for i in
                            [0, 1, 2]]

        return train, val, test
