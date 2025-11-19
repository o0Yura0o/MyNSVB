import argparse
import os
import sys
import shutil
import subprocess
from datetime import datetime
from typing import Dict, List
import numpy as np

# --- Project imports: use the original code paths ---
from utils.hparams import set_hparams, hparams
from data_gen.singing.binarize_para import SaveSpkEmb, PopBuTFyENSpkEMBinarizer


# ----- Adapters that only override metadata loading -----
class RawSaveSpkEmb(SaveSpkEmb):
    def __init__(self, item2wavfn: Dict[str, str]):
        super().__init__()
        self.item2wavfn = item2wavfn
        self.item_names = list(item2wavfn.keys())
        self._train_item_names, self._test_item_names = [], []

    def load_meta_data(self):
        # 不做任何 dataset token 過濾，以免和 YAML 中的 datasets 不一致
        self._train_item_names = []
        self._test_item_names = list(self.item_names)

    def meta_data(self, prefix):
        if prefix == 'test':
            for item_name in self._test_item_names:
                yield item_name, self.item2wavfn[item_name], 0
        else:
            return


class RawPackBinarizer(PopBuTFyENSpkEMBinarizer):
    def __init__(self, item2wavfn, spk_emb_dir):
        super().__init__()
        self.item2wavfn = item2wavfn
        self.item_names = list(item2wavfn.keys())
        self._train_item_names, self._test_item_names = [], []
        self.spk_map = {"dummy": 0}
        self.item2spk = {k: "dummy" for k in self.item_names}
        self.spk_emb_dir = spk_emb_dir.replace('\\', '/')
        print(f"[DEBUG] RawPackBinarizer set spk_emb_data_dir to {self.spk_emb_dir}")

    def load_meta_data(self):
        # 不做任何 dataset token 過濾，只建立 Amateur 名單
        self._test_item_names = [n for n in self.item_names if 'Amateur' in n]
        self._train_item_names = []

    def meta_data(self, prefix):
        if prefix != 'test':
            return
        # 只輸出 Amateur 為 key
        for name in self._test_item_names:
            pair = name.replace('Amateur', 'Professional')
            yield name, self.item2wavfn[name], 0, self.item2wavfn[pair], self.item_names

    def process_item(self, item_name=None, wav_fn=None, spk_id=None, profwavfn=None, item_names=None, binarization_args=None,):
        from utils.hparams import hparams
        hparams['spk_emb_data_dir'] = self.spk_emb_dir
        print(f"[DEBUG] process_item using spk_emb_data_dir = {hparams['spk_emb_data_dir']}")
        assert isinstance(wav_fn, str) and isinstance(profwavfn, str), \
            f"wav_fn/profwavfn must be file paths, got {type(wav_fn)} / {type(profwavfn)}"
        if binarization_args is None:
            binarization_args = getattr(hparams, "binarization_args", {})
        res = super().process_item(item_name, wav_fn, spk_id, profwavfn, item_names, binarization_args)
        return res

# ----- Utilities -----

def _ensure_dirs(work_dir: str):
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(os.path.join(work_dir, 'bin'), exist_ok=True)
    os.makedirs(os.path.join(work_dir, 'spk_emb'), exist_ok=True)
    os.makedirs(os.path.join(work_dir, 'results'), exist_ok=True)


def _synth_items(amateur_wav: str, professional_wav: str) -> Dict[str, str]:
    base = 'RAWTEST'
    amateur_item = f'RAW#singing#_{base}_Amateur_seg0'
    professional_item = amateur_item.replace('Amateur', 'Professional')
    return {
        amateur_item: os.path.abspath(amateur_wav),
        professional_item: os.path.abspath(professional_wav),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--amateur_wav', required=True)
    ap.add_argument('--professional_wav', required=True)

    ap.add_argument('--cfg_save_emb', required=True)
    ap.add_argument('--cfg_para_bin', required=True)
    ap.add_argument('--cfg_infer', required=True)

    ap.add_argument('--exp_name', required=True)

    ap.add_argument('--work_dir', default=None)
    ap.add_argument('--gpu', default=None, help='e.g., 0 or 0,1 (sets CUDA_VISIBLE_DEVICES)')

    ap.add_argument('--run_py', default='tasks/run.py')
    ap.add_argument('--run_extra', nargs=argparse.REMAINDER, help='Extra args passed to run.py after --')

    args = ap.parse_args()

    # CUDA devices (Windows compatible)
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    work_dir = args.work_dir or os.path.join('runs', f'raw_infer_{ts}')
    _ensure_dirs(work_dir)

    # ---------------- Stage 1: speaker embeddings ----------------
    set_hparams(args.cfg_save_emb, print_hparams=False)
    hparams['ds_workers'] = 0
    hparams['spk_emb_data_dir'] = os.path.abspath(os.path.join(work_dir, 'spk_emb'))
    hparams['datasets'] = ['RAW']
    hparams['test_prefixes'] = ['test']
    item2wavfn = _synth_items(args.amateur_wav, args.professional_wav)

    emb_binarizer = RawSaveSpkEmb(item2wavfn)
    emb_binarizer.item_names = list(item2wavfn.keys())
    emb_binarizer._test_item_names = emb_binarizer.item_names
    print('[Stage 1] SaveSpkEmb →', hparams['spk_emb_data_dir'])
    emb_binarizer.process_data(prefix='test')

    # ---------------- Stage 2: pack dataset ----------------
    set_hparams(args.cfg_para_bin, print_hparams=False)
    hparams['ds_workers'] = 0
    bdd = os.path.abspath(os.path.join(work_dir, 'bin'))
    hparams['binary_data_dir'] = bdd
    spk_emb_dir = os.path.abspath(os.path.join(work_dir, 'spk_emb'))
    hparams['spk_emb_data_dir'] = spk_emb_dir
    hparams['datasets'] = ['RAW']
    hparams['test_prefixes'] = ['test']

    pack_binarizer = RawPackBinarizer(item2wavfn, spk_emb_dir)
    print("[DEBUG] Pairs:")
    for k,v in item2wavfn.items():
        print(" ", k, "→", v)
    pack_binarizer.item_names = list(item2wavfn.keys())
    pack_binarizer._test_item_names = pack_binarizer.item_names
    print('[Stage 2] Pack →', bdd)
    pack_binarizer.process_data(prefix='test')

    src_phone_set = os.path.abspath("phone_set.json")
    dst_phone_set = os.path.join(bdd, "phone_set.json")
    try:
        shutil.copyfile(src_phone_set, dst_phone_set)
        print(f"[Stage 2] Copied phone_set.json → {dst_phone_set}")
    except Exception as e:
        print(f"[Stage 2] Warning: failed to copy phone_set.json: {e}")

    dummy_train_path = os.path.join(bdd, "train_lengths.npy")
    if not os.path.exists(dummy_train_path):
        np.save(dummy_train_path, np.array([0]))
        print(f"[Stage 2] Created dummy train_lengths.npy → {dummy_train_path}")

    test_f0s_path = os.path.join(bdd, "test_f0s_mean_std.npy")
    train_f0s_path = os.path.join(bdd, "train_f0s_mean_std.npy")

    if os.path.exists(test_f0s_path):
        try:
            shutil.copy(test_f0s_path, train_f0s_path)
            print(f"[Stage 2] Duplicated f0 stats → {train_f0s_path}")
        except Exception as e:
            print(f"[Stage 2] Failed to copy f0 stats: {e}")
    else:
        print("[Stage 2] Warning: test_f0s_mean_std.npy not found; train_f0s_mean_std.npy not created.")

    # ---------------- Stage 3: inference (run.py) ----------------
    set_hparams(args.cfg_infer, print_hparams=False)
    # override binary_data_dir before calling run.py
    orig_yaml = os.path.abspath(args.cfg_infer)
    yaml_dir = os.path.dirname(orig_yaml)
    custom_cfg = os.path.join(yaml_dir, "vae_global_mle_eng_temp.yaml")
    custom_bin = os.path.abspath(os.path.join(work_dir, 'bin'))
    import yaml
    with open(orig_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["binary_data_dir"] = custom_bin.replace("\\", "/")

    with open(custom_cfg, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, sort_keys=False, allow_unicode=True)

    run_cmd = [
        sys.executable, '-m', 'tasks.run',
        '--config', custom_cfg,
        '--exp_name', args.exp_name,
        '--reset', '--infer',
    ]
    if args.run_extra:
        run_cmd.extend(args.run_extra)

    print('[Stage 3] Inference via run.py →', ' '.join(run_cmd))
    print(f"[Stage 3] Using binary_data_dir override: {custom_bin}")
    proc = subprocess.run(run_cmd, check=False)
    if proc.returncode != 0:
        print('[Error] run.py returned non-zero exit status:', proc.returncode)
        sys.exit(proc.returncode)

    # Summary
    print('\n[OK] Inference from raw inputs finished')
    print('  Work dir:           ', work_dir)
    print('  Speaker embeddings: ', os.path.join(work_dir, 'spk_emb'))
    print('  Packed dataset:     ', os.path.join(work_dir, 'bin'))
    print('  (Task-defined) results dir may be under your experiment folder (per cfg_infer).')


if __name__ == '__main__':
    main()