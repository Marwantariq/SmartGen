import pickle
import os

# ---------- helper ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

os.makedirs(DATA_DIR, exist_ok=True)


def Transsas(dataset, new_env, threshold, method, model):
    with open(
        f'filter_data/{dataset}/{new_env}/{dataset}_{new_env}_generation_{method}_th={threshold}_{model}_seq_filter_true.pkl',
        'rb'
    ) as file2:
        actions = pickle.load(file2)

    out_path = os.path.join(
        DATA_DIR,
        f'{dataset}_{new_env}_generation_{method}_th={threshold}_{model}_seq.txt'
    )

    with open(out_path, 'w') as file:
        for t, behavior_list in enumerate(actions):
            for b in behavior_list:
                file.write(f"{t} {b}\n")


def Transsas_baseline(dataset, ori_env):
    with open(f'IoT_data/{dataset}/{ori_env}/trn.pkl', 'rb') as file2:
        actions = pickle.load(file2)

    out_path = os.path.join(DATA_DIR, f'{dataset}_{ori_env}_trn.txt')

    with open(out_path, 'w') as file:
        for t, behavior_list in enumerate(actions):
            for b in behavior_list:
                file.write(f"{t} {b}\n")


def Transsas_testdata(dataset, new_env):
    with open(f'IoT_data/{dataset}/{new_env}/split_test.pkl', 'rb') as file2:
        actions = pickle.load(file2)

    out_path = os.path.join(DATA_DIR, f'{dataset}_{new_env}_split_test.txt')

    with open(out_path, 'w') as file:
        for t, behavior_list in enumerate(actions):
            for b in behavior_list:
                file.write(f"{t} {b}\n")
