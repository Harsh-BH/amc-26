"""Offline tests for sm_train.py / sm_launch.py. No AWS calls, no framework: `python test_sm.py`."""
import argparse, io, json, os, pathlib, pickle, subprocess, sys, tarfile, tempfile, textwrap
from unittest import mock

import boto3
from botocore.exceptions import ClientError

import sm_launch
from sm_train import parse_params

ROOT = pathlib.Path(__file__).parent
PY = sys.executable
passed = []


def check(name):
    def deco(fn):
        fn(); passed.append(name); print(f'  ok  {name}')
    return deco


def run_train(solution_src, args=(), with_test=True, hp_json=None):
    """Run sm_train.py in a subprocess inside a fake /opt/ml, exactly as the container would."""
    tmp = pathlib.Path(tempfile.mkdtemp())
    code = tmp / 'code'; code.mkdir()
    (code / 'sm_train.py').write_text((ROOT / 'sm_train.py').read_text())
    if solution_src is not None:
        (code / 'solution.py').write_text(textwrap.dedent(solution_src))
    dirs = {k: tmp / k for k in ('train', 'test', 'model', 'out')}
    for d in dirs.values():
        d.mkdir()
    (dirs['train'] / 'train.csv').write_text('a,y\n1,10\n2,20\n3,30\n')
    if with_test:
        (dirs['test'] / 'test.csv').write_text('a\n4\n5\n')
    else:
        dirs['test'].rmdir()
    cfg = tmp / 'config'; cfg.mkdir()
    if hp_json is not None:
        (cfg / 'hyperparameters.json').write_text(json.dumps(hp_json))
    env = dict(os.environ, SM_CHANNEL_TRAIN=str(dirs['train']), SM_CHANNEL_TEST=str(dirs['test']),
               SM_MODEL_DIR=str(dirs['model']), SM_OUTPUT_DATA_DIR=str(dirs['out']),
               SM_INPUT_CONFIG_DIR=str(cfg))
    r = subprocess.run([PY, str(code / 'sm_train.py'), *args], env=env, capture_output=True, text=True)
    return r, dirs


FIT_PREDICT = '''
    import pandas as pd
    def fit(train_dir, params):
        df = pd.read_csv(train_dir / 'train.csv')
        print(f"smape={12.5}")
        return {'mean': float(df.y.mean()), 'params': params}
    def predict(model, test_dir):
        df = pd.read_csv(test_dir / 'test.csv')
        return pd.DataFrame({'a': df.a, 'y': model['mean']})
'''


def args(**kw):
    base = dict(name='job-1', region='ap-south-1', instance='ml.g4dn.xlarge', version='2.9.0',
                tag=sm_launch.DEFAULT_TAG, volume=50, max_run=3600, max_wait=7200, spot=True,
                hp={}, prefix='amc26')
    base.update(kw)
    return argparse.Namespace(**base)


ROLE = 'arn:aws:iam::123456789012:role/AMCSageMakerExecutionRole'
SESS = boto3.Session(region_name='ap-south-1', aws_access_key_id='x', aws_secret_access_key='x')

print('sm_train')


@check('params: types, dashes, bare flags')
def _():
    p = parse_params(['--epochs', '3', '--lr', '0.001', '--use-amp', '--name', 'bge', '--dims', '[1,2]', '--off', 'false'])
    assert p == {'epochs': 3, 'lr': 0.001, 'use_amp': True, 'name': 'bge', 'dims': [1, 2], 'off': False}, p
    assert parse_params(['stray', '--x', '1']) == {'x': 1}
    assert parse_params(['--neg', '-1']) == {'neg': -1}, 'negative numbers are values, not flags'


@check('fit/predict writes model.pkl + submission.csv, passes params + checkpoint_dir')
def _():
    r, d = run_train(FIT_PREDICT, ['--epochs', '3'])
    assert r.returncode == 0, r.stderr
    model = pickle.loads((d['model'] / 'model.pkl').read_bytes())
    assert model['mean'] == 20.0 and model['params']['epochs'] == 3
    assert 'checkpoint_dir' in model['params']
    sub = (d['out'] / 'submission.csv').read_text().splitlines()
    assert sub == ['a,y', '4,20.0', '5,20.0'], sub
    assert 'smape=12.5' in r.stdout, 'metric line must reach stdout for MetricDefinitions'


@check('main() path gets every directory')
def _():
    r, d = run_train('''
        def main(train_dir, test_dir, model_dir, out_dir, params):
            (model_dir / 'w.txt').write_text(str(sorted(p.name for p in train_dir.iterdir())))
            (out_dir / 'o.txt').write_text(str(params['k']))
    ''', ['--k', '7'])
    assert r.returncode == 0, r.stderr
    assert (d['model'] / 'w.txt').read_text() == "['train.csv']"
    assert (d['out'] / 'o.txt').read_text() == '7'


@check('no test channel: fit runs, predict is skipped cleanly')
def _():
    r, d = run_train(FIT_PREDICT, with_test=False)
    assert r.returncode == 0, r.stderr
    assert (d['model'] / 'model.pkl').exists() and not (d['out'] / 'submission.csv').exists()


@check('fit returning None: no model.pkl written')
def _():
    r, d = run_train('def fit(t, p):\n    return None\n')
    assert r.returncode == 0 and not (d['model'] / 'model.pkl').exists()


@check('missing solution.py fails loudly')
def _():
    r, _ = run_train(None)
    assert r.returncode != 0 and 'no solution.py' in r.stderr


@check('solution importing a missing package surfaces the REAL error, not "no solution.py"')
def _():
    r, _ = run_train('import definitely_not_installed_pkg\ndef fit(t, p): pass\n')
    assert r.returncode != 0
    assert 'definitely_not_installed_pkg' in r.stderr and 'no solution.py' not in r.stderr, r.stderr


@check('solution with neither main nor fit fails loudly')
def _():
    r, _ = run_train('x = 1\n')
    assert r.returncode != 0 and 'neither main() nor fit()' in r.stderr


@check('exception inside fit propagates as non-zero exit (SageMaker marks job Failed)')
def _():
    r, _ = run_train('def fit(t, p):\n    raise RuntimeError("boom")\n')
    assert r.returncode != 0 and 'boom' in r.stderr


@check('hyperparameters.json round-trip is lossless: false, lists, null, numeric strings, dicts')
def _():
    hp_in = {'use_amp': False, 'flag_on': True, 'dims': [128, 256], 'none_val': None,
             'code': '3', 'lr': 0.001, 'cfg': {'a': 1, 'b': [2]}, 'name': 'bge-m3'}
    with mock.patch.object(sm_launch, 'image_uri', lambda *a: 'img'):
        req = sm_launch.build_request(args(hp=hp_in), ROLE, 's3://b/s.tgz', 's3://bkt/tr/', None, 'bkt')
    r, d = run_train('''
        import json
        def fit(t, params):
            params.pop('checkpoint_dir')
            print('PARAMS=' + json.dumps(params, sort_keys=True))
    ''', args=['--use_amp', 'False', '--dims', '[128,'], hp_json=req['HyperParameters'])
    assert r.returncode == 0, r.stderr
    got = json.loads(r.stdout.split('PARAMS=')[1].splitlines()[0])
    assert got == hp_in, f'lossy: {got}'


@check('without hyperparameters.json (local run), argv is used')
def _():
    r, _ = run_train('def fit(t, p):\n    print("EPOCHS", repr(p["epochs"]))\n', args=['--epochs', '4'])
    assert r.returncode == 0 and 'EPOCHS 4' in r.stdout, r.stdout + r.stderr


print('sm_launch')


@check('pack_source: ships top-level .py + requirements, never itself or tests')
def _():
    blob, names = sm_launch.pack_source()
    members = tarfile.open(fileobj=io.BytesIO(blob)).getnames()
    assert members == names
    assert 'sm_train.py' in names and 'requirements.txt' in names
    assert 'sm_launch.py' not in names and not any(n.startswith('test_') for n in names)
    assert all('/' not in n for n in names), 'no directories (.venv, data) may leak in'


@check('pack_source: --solution ships the file AS solution.py, not under its own name')
def _():
    alt = ROOT / 'solution_alt_tmp.py'
    alt.write_text('def fit(t, p): return 1\n')
    try:
        blob, names = sm_launch.pack_source(str(alt))
        t = tarfile.open(fileobj=io.BytesIO(blob))
        assert 'solution.py' in names and 'solution_alt_tmp.py' not in names, names
        assert t.extractfile('solution.py').read() == alt.read_bytes()
    finally:
        alt.unlink()


@check('pack_source: requirements.txt is sm_requirements.txt, not the local env')
def _():
    blob, _ = sm_launch.pack_source()
    got = tarfile.open(fileobj=io.BytesIO(blob)).extractfile('requirements.txt').read()
    assert got == (ROOT / 'sm_requirements.txt').read_bytes()
    import re
    reqs = {re.split(r'[<>=!~\[; ]', l.strip())[0].lower() for l in got.decode().splitlines()
            if l.strip() and not l.lstrip().startswith('#')}
    preinstalled = {'torch', 'torchvision', 'numpy', 'pandas', 'scikit-learn', 'scipy', 'pillow', 'boto3'}
    assert reqs and not reqs & preinstalled, f're-resolving {reqs & preinstalled} wastes job start time'
    assert 'lightgbm' in reqs


with mock.patch.object(sm_launch, 'image_uri', lambda region, inst, ver, tag: f'img:{inst}'):
    @check('spot: sets flag, MaxWait >= MaxRuntime, per-job checkpoint prefix')
    def _():
        r = sm_launch.build_request(args(max_run=9000, max_wait=100), ROLE, 's3://bkt/src.tgz', 's3://bkt/tr/', None, 'bkt')
        assert r['EnableManagedSpotTraining'] is True
        assert r['StoppingCondition'] == {'MaxRuntimeInSeconds': 9000, 'MaxWaitTimeInSeconds': 9000}
        assert r['CheckpointConfig']['S3Uri'].endswith('/checkpoints/job-1/')
        assert sm_launch.validate(SESS, r)

    @check('no-spot: no spot keys at all')
    def _():
        r = sm_launch.build_request(args(spot=False), ROLE, 's3://bkt/src.tgz', 's3://bkt/tr/', None, 'bkt')
        assert not {'EnableManagedSpotTraining', 'CheckpointConfig'} & set(r)
        assert 'MaxWaitTimeInSeconds' not in r['StoppingCondition']
        assert sm_launch.validate(SESS, r)

    @check('test channel only when a test source is given')
    def _():
        without = sm_launch.build_request(args(), ROLE, 's3://bkt/src.tgz', 's3://bkt/tr/', None, 'bkt')
        with_ = sm_launch.build_request(args(), ROLE, 's3://bkt/src.tgz', 's3://bkt/tr/', 's3://bkt/te/', 'bkt')
        assert [c['ChannelName'] for c in without['InputDataConfig']] == ['train']
        assert [c['ChannelName'] for c in with_['InputDataConfig']] == ['train', 'test']
        assert sm_launch.validate(SESS, with_)

    @check('hyperparameters: every value JSON-encoded (strings too), metrics defined')
    def _():
        hp_in = {'epochs': 3, 'lr': 0.001, 'm': 'bge', 'amp': True}
        r = sm_launch.build_request(args(hp=hp_in), ROLE, 's3://b/s.tgz', 's3://bkt/tr/', None, 'bkt')
        hp = r['HyperParameters']
        assert hp == {'sagemaker_program': '"sm_train.py"', 'sagemaker_submit_directory': '"s3://b/s.tgz"',
                      'epochs': '3', 'lr': '0.001', 'm': '"bge"', 'amp': 'true'}, hp
        assert all(isinstance(v, str) for v in hp.values()), 'the API requires string values'
        names = [m['Name'] for m in r['AlgorithmSpecification']['MetricDefinitions']]
        assert names == ['smape', 'val_loss']


@check('metric regexes match what solution.py prints, and nothing else')
def _():
    import re
    rx = {m['Name']: re.compile(m['Regex']) for m in sm_launch.METRICS}
    assert rx['smape'].search('epoch 3 smape=12.34').group(1) == '12.34'
    assert rx['smape'].search('smape=1.5e-3').group(1) == '1.5e-3'
    assert rx['val_loss'].search('val_loss=0.25 lr=1').group(1) == '0.25'
    assert rx['smape'].search('holdout_smape=9') is None, 'must not match inside another name'


@check('image: SDK resolves cpu vs gpu; bad version falls back to pinned tag')
def _():
    gpu = sm_launch.image_uri('ap-south-1', 'ml.g4dn.xlarge', '2.9.0')
    cpu = sm_launch.image_uri('ap-south-1', 'ml.m5.large', '2.9.0')
    assert gpu.endswith('pytorch-training:2.9.0-gpu-py312') and cpu.endswith('pytorch-training:2.9.0-cpu-py312')
    assert sm_launch.image_uri('ap-south-1', 'ml.m5.large', '99.0.0').endswith(':2.9-cpu-py312')


@check('validate rejects a malformed request')
def _():
    try:
        sm_launch.validate(SESS, {'TrainingJobName': 'x', 'ResourceConfig': {'VolumeSizeInGB': 'thirty'}})
    except SystemExit as e:
        assert 'VolumeSizeInGB' in str(e) and 'RoleArn' in str(e)
    else:
        raise AssertionError('should have rejected')


class FakeQuotas:
    def __init__(self, value, default):
        self.value, self.default = value, default
    def get_paginator(self, _):
        v = self.value
        return mock.Mock(paginate=lambda **k: [{'Quotas': [
            {'QuotaName': 'ml.m5.large for training job usage', 'Value': v, 'QuotaCode': 'L-OD'},
            {'QuotaName': 'ml.m5.large for spot training job usage', 'Value': v, 'QuotaCode': 'L-SP'}]}])
    def get_aws_default_service_quota(self, **k):
        return {'Quota': {'Value': self.default}}


@check('quota preflight: 0 -> advice asking above the AWS default; >=1 -> clear')
def _():
    sess = mock.Mock(client=lambda _: FakeQuotas(0, 4))
    val, adv = sm_launch.quota_advice(sess, 'ml.m5.large', True, 'ap-south-1', 'amc')
    assert val == 0 and '--quota-code L-SP --desired-value 5' in adv, adv
    val, adv = sm_launch.quota_advice(mock.Mock(client=lambda _: FakeQuotas(0, 0)), 'ml.m5.large', False, 'ap-south-1', 'amc')
    assert '--quota-code L-OD --desired-value 1' in adv
    assert sm_launch.quota_advice(mock.Mock(client=lambda _: FakeQuotas(2, 0)), 'ml.m5.large', False, 'r', 'p') == (2, None)
    val, adv = sm_launch.quota_advice(mock.Mock(client=lambda _: FakeQuotas(1, 0)), 'ml.x9.huge', False, 'r', 'p')
    assert val is None and 'check the instance name' in adv


@check('submit retries while a new role propagates, then gives up on other errors')
def _():
    err = lambda code, msg: ClientError({'Error': {'Code': code, 'Message': msg}}, 'CreateTrainingJob')
    sm = mock.Mock()
    sm.create_training_job.side_effect = [err('ValidationException', 'Could not assume role arn:...'),
                                          err('ValidationException', 'could not assume role'), {'ok': 1}]
    with mock.patch.object(sm_launch.time, 'sleep'):
        assert sm_launch.submit(sm, {}) == {'ok': 1}
    assert sm.create_training_job.call_count == 3
    sm = mock.Mock(); sm.create_training_job.side_effect = err('ResourceLimitExceeded', 'quota')
    try:
        sm_launch.submit(sm, {})
    except ClientError as e:
        assert e.response['Error']['Code'] == 'ResourceLimitExceeded'
    else:
        raise AssertionError('non-role errors must not be retried')
    assert sm.create_training_job.call_count == 1


@check('launch refuses (before uploading anything) without solution / data / quota')
def _():
    with mock.patch.object(sm_launch.boto3, 'Session') as S, \
         mock.patch.object(sm_launch, 'quota_advice', return_value=(0, 'QUOTA IS 0')):
        S.return_value.client.return_value.get_caller_identity.return_value = {'Account': '1'}
        try:
            sm_launch.main(['--solution', 'nope.py', '--instance', 'ml.m5.large'])
        except SystemExit as e:
            msg = str(e)
        assert 'no solution file' in msg and 'QUOTA IS 0' in msg and '--data' in msg, msg
        S.return_value.client.return_value.put_object.assert_not_called()


def _describe(status, secondary='Completed', **kw):
    d = {'TrainingJobStatus': status, 'SecondaryStatus': secondary,
         'ResourceConfig': {'InstanceType': 'ml.m5.large'}, 'EnableManagedSpotTraining': True,
         'TrainingTimeInSeconds': 300, 'BillableTimeInSeconds': 90,
         'FinalMetricDataList': [{'MetricName': 'smape', 'Value': 9.43}]}
    d.update(kw)
    return d


class FakeLogs:
    """Two pages of events, then CloudWatch's end-of-stream: empty page, same token."""
    def __init__(self):
        self.calls = 0
    def describe_log_streams(self, **k):
        return {'logStreams': [{'logStreamName': 'job/algo-1'}]}
    def get_log_events(self, **k):
        self.calls += 1
        assert self.calls < 10, 'log pagination does not terminate'
        tok = k.get('nextToken')
        if tok is None:
            return {'events': [{'message': 'line1'}, {'message': 'line2'}], 'nextForwardToken': 't1'}
        if tok == 't1':
            return {'events': [{'message': 'KeyError: price'}], 'nextForwardToken': 't2'}
        return {'events': [], 'nextForwardToken': tok}


@check('logs: follows pagination to the end and terminates')
def _():
    fl = FakeLogs()
    lines = sm_launch.logs(mock.Mock(client=lambda _: fl), 'job')
    assert lines == ['line1', 'line2', 'KeyError: price'] and fl.calls == 3
    assert sm_launch.logs(mock.Mock(client=lambda _: FakeLogs()), 'job', tail=1) == ['KeyError: price']


@check('wait: reports transitions, stops on Completed, shows metrics + billable time')
def _():
    sm = mock.Mock()
    sm.describe_training_job.side_effect = [_describe('InProgress', 'Starting'), _describe('InProgress', 'Training'),
                                            _describe('InProgress', 'Training'), _describe('Completed')]
    buf = io.StringIO()
    with mock.patch.object(sm_launch.time, 'sleep'), mock.patch('sys.stdout', buf):
        d = sm_launch.wait(mock.Mock(client=lambda _: sm), 'job')
    out = buf.getvalue()
    assert d['TrainingJobStatus'] == 'Completed' and sm.describe_training_job.call_count == 4
    assert out.count('InProgress / Training') == 1, 'unchanged states must not be re-printed'
    assert '"smape": 9.43' in out and '"BillableTimeInSeconds": 90' in out
    assert 'last 40 log lines' not in out


@check('wait: on Failed, prints FailureReason and the tail of the logs')
def _():
    sm = mock.Mock()
    sm.describe_training_job.side_effect = [_describe('Failed', 'Failed', FailureReason='AlgorithmError: KeyError')]
    clients = {'sagemaker': sm, 'logs': FakeLogs()}
    buf = io.StringIO()
    with mock.patch('sys.stdout', buf):
        sm_launch.wait(mock.Mock(client=lambda n: clients[n]), 'job')
    out = buf.getvalue()
    assert 'AlgorithmError: KeyError' in out and 'KeyError: price' in out


def _tgz(files):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w:gz') as t:
        for name, data in files.items():
            info = tarfile.TarInfo(name); info.size = len(data); t.addfile(info, io.BytesIO(data))
    return buf.getvalue()


@check('fetch: extracts model.tar.gz AND output.tar.gz (submission.csv lives in the second)')
def _():
    import shutil
    name = 'test-fetch-tmp'
    objs = {'amc26/output/j/output/model.tar.gz': _tgz({'model.pkl': b'M'}),
            'amc26/output/j/output/output.tar.gz': _tgz({'submission.csv': b'id,y\n1,2\n'})}
    s3 = mock.Mock(download_file=lambda b, k, dst: pathlib.Path(dst).write_bytes(objs[k]))
    sm = mock.Mock(describe_training_job=lambda **k: _describe(
        'Completed', ModelArtifacts={'S3ModelArtifacts': 's3://bkt/amc26/output/j/output/model.tar.gz'}))
    clients = {'sagemaker': sm, 's3': s3}
    try:
        dest = sm_launch.fetch(mock.Mock(client=lambda n: clients[n]), name)
        assert (dest / 'model.pkl').read_bytes() == b'M'
        assert (dest / 'submission.csv').read_text() == 'id,y\n1,2\n'
        assert not list(dest.glob('*.tar.gz')), 'archives are removed after extraction'
    finally:
        shutil.rmtree(ROOT / 'out' / name, ignore_errors=True)


@check('fetch: missing output.tar.gz is tolerated; a non-Completed job is refused')
def _():
    import shutil
    name = 'test-fetch-tmp2'
    def dl(b, k, dst):
        if k.endswith('output.tar.gz'):
            raise ClientError({'Error': {'Code': '404', 'Message': 'nf'}}, 'HeadObject')
        pathlib.Path(dst).write_bytes(_tgz({'model.pkl': b'M'}))
    sm = mock.Mock(describe_training_job=lambda **k: _describe(
        'Completed', ModelArtifacts={'S3ModelArtifacts': 's3://bkt/p/model.tar.gz'}))
    clients = {'sagemaker': sm, 's3': mock.Mock(download_file=dl)}
    try:
        with mock.patch('sys.stdout', io.StringIO()):
            dest = sm_launch.fetch(mock.Mock(client=lambda n: clients[n]), name)
        assert (dest / 'model.pkl').exists()
    finally:
        shutil.rmtree(ROOT / 'out' / name, ignore_errors=True)
    sm = mock.Mock(describe_training_job=lambda **k: _describe('Failed', 'Failed'))
    try:
        sm_launch.fetch(mock.Mock(client=lambda n: sm), 'x')
    except SystemExit as e:
        assert 'Failed' in str(e)
    else:
        raise AssertionError('must refuse to fetch a failed job')


print(f'\n{len(passed)} passed')
