"""SageMaker entry point. Problem-agnostic: it owns the SageMaker filesystem contract,
your `solution.py` owns the modelling.

Inside SageMaker the container gets:
  /opt/ml/input/data/train, /opt/ml/input/data/test   the channels   (SM_CHANNEL_TRAIN / _TEST)
  /opt/ml/model         tarred to model.tar.gz when the job ends      (SM_MODEL_DIR)
  /opt/ml/output/data   tarred to output.tar.gz                       (SM_OUTPUT_DATA_DIR)
  /opt/ml/checkpoints   synced to S3 continuously - survives a spot interruption
  hyperparameters in /opt/ml/input/config/hyperparameters.json (read directly - see load_params)

Run it locally with the same code path: `python sm_train.py --epochs 3` reads data/train and
data/test and writes to out/local/ when /opt/ml does not exist.

solution.py exposes either
    fit(train_dir: Path, params: dict) -> model        (pickled to model.pkl unless None)
    predict(model, test_dir: Path) -> DataFrame        (optional, -> submission.csv)
or  main(train_dir, test_dir, model_dir, out_dir, params) for full control.

params always carries 'checkpoint_dir'. Write checkpoints there and resume from it if present.
Print `smape=<float>` / `val_loss=<float>` and SageMaker records them as job metrics.
"""
import json, os, pathlib, pickle, sys, time

ON_SAGEMAKER = pathlib.Path('/opt/ml').exists()
_LOCAL = pathlib.Path(__file__).parent
TRAIN = pathlib.Path(os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train' if ON_SAGEMAKER else _LOCAL / 'data/train'))
TEST = pathlib.Path(os.environ.get('SM_CHANNEL_TEST', '/opt/ml/input/data/test' if ON_SAGEMAKER else _LOCAL / 'data/test'))
MODEL_DIR = pathlib.Path(os.environ.get('SM_MODEL_DIR', '/opt/ml/model' if ON_SAGEMAKER else _LOCAL / 'out/local/model'))
OUT_DIR = pathlib.Path(os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/output/data' if ON_SAGEMAKER else _LOCAL / 'out/local/output'))
CKPT_DIR = pathlib.Path('/opt/ml/checkpoints' if ON_SAGEMAKER else _LOCAL / 'out/local/checkpoints')
HP_FILE = pathlib.Path(os.environ.get('SM_INPUT_CONFIG_DIR', '/opt/ml/input/config')) / 'hyperparameters.json'


def parse_params(argv):
    """--key value pairs; values JSON-decoded so '3' -> 3, 'true' -> True. Bare --flag -> True."""
    params, i = {}, 0
    while i < len(argv):
        if not argv[i].startswith('--'):
            i += 1
            continue
        key = argv[i][2:].replace('-', '_')
        if i + 1 < len(argv) and not argv[i + 1].startswith('--'):
            raw, i = argv[i + 1], i + 2
            try:
                params[key] = json.loads(raw)
            except ValueError:
                params[key] = raw
        else:
            params[key], i = True, i + 1
    return params


def load_params(argv):
    """On SageMaker read hyperparameters.json, never argv. The toolkit rebuilds argv with str(),
    which turns false into 'False' (truthy), splits [128, 256] on the space, and null into a bare
    flag. The JSON file holds exactly what sm_launch sent: every value JSON-encoded."""
    if HP_FILE.exists():
        params = {}
        for k, v in json.loads(HP_FILE.read_text()).items():
            if k.startswith('sagemaker_'):
                continue
            try:
                params[k] = json.loads(v)
            except (ValueError, TypeError):
                params[k] = v            # sent by something other than sm_launch
        return params
    return parse_params(argv)


def _ls(p):
    if not p.exists():
        return 'ABSENT'
    items = sorted(x.name for x in p.iterdir())
    return f'{len(items)} entries {items[:6]}{" ..." if len(items) > 6 else ""}'


def main(argv=None):
    t0 = time.time()
    params = load_params(sys.argv[1:] if argv is None else argv)
    for d in (MODEL_DIR, OUT_DIR, CKPT_DIR):
        d.mkdir(parents=True, exist_ok=True)
    params.setdefault('checkpoint_dir', str(CKPT_DIR))
    print(f'[sm_train] {"sagemaker" if ON_SAGEMAKER else "local"} | params={params}', flush=True)
    print(f'[sm_train] train: {_ls(TRAIN)}', flush=True)
    print(f'[sm_train] test:  {_ls(TEST)}', flush=True)

    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    try:
        import solution
    except ModuleNotFoundError as e:
        if e.name != 'solution':
            raise                       # solution.py exists but imports something missing
        sys.exit('[sm_train] no solution.py next to sm_train.py - nothing to run')

    if hasattr(solution, 'main'):
        solution.main(TRAIN, TEST, MODEL_DIR, OUT_DIR, params)
    elif hasattr(solution, 'fit'):
        model = solution.fit(TRAIN, params)
        if model is not None:
            with open(MODEL_DIR / 'model.pkl', 'wb') as f:
                pickle.dump(model, f)
        if hasattr(solution, 'predict') and TEST.exists():
            sub = solution.predict(model, TEST)
            sub.to_csv(OUT_DIR / 'submission.csv', index=False)
            print(f'[sm_train] wrote submission.csv ({len(sub)} rows)', flush=True)
    else:
        sys.exit('[sm_train] solution.py defines neither main() nor fit()')
    print(f'[sm_train] done in {time.time() - t0:.1f}s', flush=True)


if __name__ == '__main__':
    main()
