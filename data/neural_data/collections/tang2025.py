from torch.utils.data import ConcatDataset
from .. import Assembly, TemporalAssemblyDataset, get_data_loader
from ..utils import compute_joint_ceiling

CLIP_DATASETS = [
    "McMahon2023-fMRI",
    "Lahner2024-fMRI",
]

MOVIE_DATASETS = [
    'Keles2024-fMRI',
    'Berezutskaya2021-fMRI',
    'Savasegal2023-fMRI-Defeat',
    'Savasegal2023-fMRI-Growth',
    'Savasegal2023-fMRI-Iteration',
    'Savasegal2023-fMRI-Lemonade',
]

clip_assemblies = [Assembly(assembly_name=name) for name in CLIP_DATASETS]
movie_assemblies = [Assembly(assembly_name=name) for name in MOVIE_DATASETS]
all_assemblies = clip_assemblies + movie_assemblies

def get_datasets(video_transform, type='all', fps=12):
    datasets = []
    dataset_names = []
    
    if type in ['all', 'clip']:
        clip_datasets = [
            TemporalAssemblyDataset(
                assembly, 
                fps=fps, 
                context_duration=3000,
                response_delay_duration=0, 
                transform=video_transform) 
                for assembly in clip_assemblies
            ]
        datasets += clip_datasets
        dataset_names += CLIP_DATASETS

    if type in ['all', 'movie']:
        movie_datasets = [
            TemporalAssemblyDataset(
                assembly, 
                fps=fps, 
                context_duration=3000, 
                response_delay_duration=5000, 
                transform=video_transform) 
                for assembly in movie_assemblies
            ]
        datasets += movie_datasets
        dataset_names += MOVIE_DATASETS

    datasets = {
        name: dataset 
        for name, dataset in zip(dataset_names, datasets)
    }

    return datasets

def get_compilation(video_transform, type='all', ratios=(0.8, 0.1, 0.1), clip_duration=60, fps=12, return_ceiling=False, seed=42):

    datasets = get_datasets(video_transform=video_transform, fps=fps, type=type)

    trainsets = []
    valsets = []
    testsets = []
    for dataset in datasets.values():
        tr, va, te = dataset.subset(ratios, time_bin_block=clip_duration, seed=seed)
        trainsets.append(tr)
        valsets.append(va)
        testsets.append(te)
    trainset = ConcatDataset(trainsets) 
    valset = ConcatDataset(valsets)
    testset = ConcatDataset(testsets)

    if return_ceiling:
        ceiling = compute_joint_ceiling([d.assembly for d in datasets.values()])
        return trainset, valset, testset, ceiling

    return trainset, valset, testset