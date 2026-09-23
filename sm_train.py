"""SageMaker entry point. Problem-agnostic: it owns the SageMaker filesystem contract,
your `solution.py` owns the modelling.

SageMaker hands the container:
  /opt/ml/input/data/<channel>   the data you uploaded      (env SM_CHANNEL_<CHANNEL>)
  /opt/ml/model                  tarred to S3 when the job ends   (env SM_MODEL_DIR)
  /opt/ml/output/data            also tarred to S3           (env SM_OUTPUT_DATA_DIR)
  hyperparameters as --key value CLI args

Write a solution.py next to this file exposing either:
    fit(train_dir: Path, params: dict) -> model
    predict(model, test_dir: Path) -> pandas.DataFrame   # optional
or a single main(train_dir, test_dir, model_dir, out_dir, params) for full control.
"""
import argparse, json, os, pathlib, pickle, sys, time

TRAIN = pathlib.Path(os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train'))
TEST = pathlib.Path(os.environ.get('SM_CHANNEL_TEST', '/opt/ml/input/data/test'))
MODEL_DIR = pathlib.Path(os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
OUT_DIR = pathlib.Path(os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/output/data'))


def parse_params():
    """SageMaker passes hyperparameters as --key value. Accept anything."""
    ap = argparse.ArgumentParser()
    known, rest = ap.parse_known_args()
    params, i = {}, 0
    while i < len(rest):
        if rest[i].startswith('--'):
            k = rest[i][2:]
            v = rest[i + 1] if i + 1 < len(rest) and not rest[i + 1].startswith('--') else 'true'
            try:
                v = json.loads(v)          # numbers, bools, lists arrive as strings
            except (ValueError, TypeError):
                pass
            params[k] = v
            i += 2
        else:
            i += 1
    return params


def main():
    t0 = time.time()
    params = parse_params()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f'[sm_train] params={params}', flush=True)
    for label, p in (('train', TRAIN), ('test', TEST)):
        print(f'[sm_train] {label}: {sorted(x.name for x in p.iterdir()) if p.exists() else "ABSENT"}', flush=True)

    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    try:
        import solution
    except ImportError:
        sys.exit('[sm_train] no solution.py found next to sm_train.py - nothing to run')

    if hasattr(solution, 'main'):
        solution.main(TRAIN, TEST, MODEL_DIR, OUT_DIR, params)
    else:
        model = solution.fit(TRAIN, params)
        with open(MODEL_DIR / 'model.pkl', 'wb') as f:
            pickle.dump(model, f)
        if hasattr(solution, 'predict') and TEST.exists():
            solution.predict(model, TEST).to_csv(OUT_DIR / 'submission.csv', index=False)
            print('[sm_train] wrote submission.csv', flush=True)

    # Printed so SageMaker metric regexes and the console log can pick it up.
    print(f'[sm_train] done in {time.time() - t0:.1f}s', flush=True)


if __name__ == '__main__':
    main()
