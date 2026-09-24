"""Submit and manage SageMaker training jobs. boto3 builds and submits the request; the
sagemaker SDK is used only to resolve the DLC image URI (falls back to a pinned tag).

    python sm_launch.py --create-role                           # once per account
    python sm_launch.py --check                                 # role, bucket, quota for --instance
    python sm_launch.py --data data/train --test-data data/test --local    # real container, free
    python sm_launch.py --data data/train --test-data data/test --dry-run
    python sm_launch.py --data data/train --test-data data/test --wait
    python sm_launch.py --status JOB | --logs JOB | --fetch JOB    # fetch -> out/JOB/

Packs every top-level *.py (except this file and test_*.py) plus sm_requirements.txt, which
the container pip-installs as requirements.txt. --solution FILE ships FILE as solution.py,
so experiments can live side by side (solution_v2.py, ...).

Spot is on by default (~70% cheaper). Checkpoint to params['checkpoint_dir'] in solution.py
for anything over ~1h - an interruption otherwise restarts from zero.
"""
import argparse, io, json, os, pathlib, sys, tarfile, time
from concurrent.futures import ThreadPoolExecutor
import boto3
from botocore.exceptions import ClientError

ROOT = pathlib.Path(__file__).parent
DLC_ACCOUNT = '763104351884'          # serves ap-south-1; verified against sagemaker-core config
DEFAULT_TAG = '2.9-gpu-py312'         # fallback only, when the SDK lookup fails
ROLE_NAME = 'AMCSageMakerExecutionRole'
CPU_PREFIXES = ('ml.m', 'ml.c', 'ml.t', 'ml.r')
METRICS = [{'Name': 'smape', 'Regex': r'\bsmape=([0-9.eE+-]+)'},
           {'Name': 'val_loss', 'Regex': r'\bval_loss=([0-9.eE+-]+)'}]
TERMINAL = ('Completed', 'Failed', 'Stopped')


def is_cpu(instance):
    return instance.startswith(CPU_PREFIXES)


def image_uri(region, instance, version, tag=DEFAULT_TAG):
    try:
        from sagemaker.core import image_uris
        return image_uris.retrieve('pytorch', region=region, version=version, instance_type=instance,
                                   image_scope='training', py_version='py312')
    except Exception as e:
        print(f'  [image] SDK lookup failed ({type(e).__name__}), using pinned tag', file=sys.stderr)
    t = tag.replace('-gpu-', '-cpu-') if is_cpu(instance) else tag
    return f'{DLC_ACCOUNT}.dkr.ecr.{region}.amazonaws.com/pytorch-training:{t}'


# ---------------------------------------------------------------- account setup

def create_role(sess):
    iam = sess.client('iam')
    trust = {'Version': '2012-10-17', 'Statement': [{'Effect': 'Allow',
             'Principal': {'Service': 'sagemaker.amazonaws.com'}, 'Action': 'sts:AssumeRole'}]}
    try:
        arn = iam.create_role(RoleName=ROLE_NAME, AssumeRolePolicyDocument=json.dumps(trust),
                              Description='Amazon ML Challenge training jobs')['Role']['Arn']
        print(f'created {arn}')
    except iam.exceptions.EntityAlreadyExistsException:
        arn = iam.get_role(RoleName=ROLE_NAME)['Role']['Arn']
        print(f'exists  {arn}')
    # SageMakerFullAccess only grants S3 on buckets named *sagemaker*; S3FullAccess covers --bucket.
    for p in ('AmazonSageMakerFullAccess', 'AmazonS3FullAccess'):
        iam.attach_role_policy(RoleName=ROLE_NAME, PolicyArn=f'arn:aws:iam::aws:policy/{p}')
    print('attached AmazonSageMakerFullAccess + AmazonS3FullAccess')
    return arn


def role_arn(sess, dry_run=False):
    try:
        return sess.client('iam').get_role(RoleName=ROLE_NAME)['Role']['Arn']
    except ClientError:
        if dry_run:
            return f'arn:aws:iam::000000000000:role/{ROLE_NAME}'
        sys.exit(f'no role {ROLE_NAME}. Run: python sm_launch.py --create-role')


def quota(sess, instance, spot):
    """(value, code) for '<instance> for [spot ]training job usage'. New accounts are often 0."""
    want = f'{instance} for {"spot " if spot else ""}training job usage'
    sq = sess.client('service-quotas')
    for page in sq.get_paginator('list_service_quotas').paginate(ServiceCode='sagemaker'):
        for q in page['Quotas']:
            if q['QuotaName'] == want:
                return q['Value'], q['QuotaCode']
    return None, None


def quota_advice(sess, instance, spot, region, profile):
    val, code = quota(sess, instance, spot)
    kind = 'spot training' if spot else 'training'
    if val is None:
        return None, f'no {kind} quota found for {instance} - check the instance name'
    if val >= 1:
        return val, None
    try:
        default = sess.client('service-quotas').get_aws_default_service_quota(
            ServiceCode='sagemaker', QuotaCode=code)['Quota']['Value']
    except ClientError:
        default = 0
    ask = max(1, int(default) + 1)   # AWS rejects requests that are not above the default
    return val, (f'{instance} {kind} quota is 0 in {region}. Request it (free, approval can take hours):\n'
                 f'  aws service-quotas request-service-quota-increase --service-code sagemaker '
                 f'--quota-code {code} --desired-value {ask} --region {region} --profile {profile}')


def ensure_bucket(sess, bucket, region):
    s3 = sess.client('s3')
    try:
        s3.head_bucket(Bucket=bucket)
    except ClientError:
        kw = {'Bucket': bucket}
        if region != 'us-east-1':
            kw['CreateBucketConfiguration'] = {'LocationConstraint': region}
        s3.create_bucket(**kw)
        print(f'created s3://{bucket}')


# ---------------------------------------------------------------- packaging + upload

def pack_source(solution=None):
    """Return (tar.gz bytes, member names). Top-level *.py, solution override, requirements."""
    files = {p.name: p for p in sorted(ROOT.glob('*.py'))
             if p.name != 'sm_launch.py' and not p.name.startswith('test_')}
    if solution:
        sol = pathlib.Path(solution)
        if not sol.exists():
            sys.exit(f'--solution {solution} not found')
        files.pop(sol.name, None)            # don't also ship solution_v2.py under its own name
        files['solution.py'] = sol
    if (ROOT / 'sm_requirements.txt').exists():
        files['requirements.txt'] = ROOT / 'sm_requirements.txt'
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w:gz') as tar:
        for arc in sorted(files):            # sorted -> same inputs, same tarball
            tar.add(files[arc], arcname=arc)
    return buf.getvalue(), sorted(files)


def upload_dir(sess, local, bucket, prefix):
    local = pathlib.Path(local)
    if not local.is_dir():
        sys.exit(f'{local} is not a directory')
    files = [f for f in sorted(local.rglob('*')) if f.is_file() and not f.name.startswith('.')]
    if not files:
        sys.exit(f'{local} has no files')
    s3 = sess.client('s3')
    with ThreadPoolExecutor(16) as ex:
        list(ex.map(lambda f: s3.upload_file(str(f), bucket, f'{prefix}/{f.relative_to(local).as_posix()}'), files))
    print(f'  {local} -> s3://{bucket}/{prefix}/ ({len(files)} files)')
    return f's3://{bucket}/{prefix}/'


# ---------------------------------------------------------------- request

def channel(name, uri):
    return {'ChannelName': name, 'DataSource': {'S3DataSource': {
        'S3DataType': 'S3Prefix', 'S3Uri': uri, 'S3DataDistributionType': 'FullyReplicated'}}}


def build_request(a, role, src_s3, train_s3, test_s3, bucket):
    # Every value JSON-encoded, strings included, exactly as the sagemaker SDK does: the toolkit
    # json-decodes them, and sm_train decodes hyperparameters.json the same way, so '3' stays a
    # string, false stays False, and [128, 256] survives intact.
    hp = {k: json.dumps(v) for k, v in
          {'sagemaker_program': 'sm_train.py', 'sagemaker_submit_directory': src_s3, **a.hp}.items()}
    req = {
        'TrainingJobName': a.name,
        'RoleArn': role,
        'AlgorithmSpecification': {'TrainingImage': image_uri(a.region, a.instance, a.version, a.tag),
                                   'TrainingInputMode': 'File', 'MetricDefinitions': METRICS},
        'HyperParameters': hp,
        'InputDataConfig': [channel('train', train_s3)] + ([channel('test', test_s3)] if test_s3 else []),
        'OutputDataConfig': {'S3OutputPath': f's3://{bucket}/{a.prefix}/output/'},
        'ResourceConfig': {'InstanceType': a.instance, 'InstanceCount': 1, 'VolumeSizeInGB': a.volume},
        'StoppingCondition': {'MaxRuntimeInSeconds': a.max_run},
        'Tags': [{'Key': 'project', 'Value': 'amc26'}],
    }
    if a.spot:
        req['EnableManagedSpotTraining'] = True
        req['StoppingCondition']['MaxWaitTimeInSeconds'] = max(a.max_wait, a.max_run)
        req['CheckpointConfig'] = {'S3Uri': f's3://{bucket}/{a.prefix}/checkpoints/{a.name}/'}
    return req


def validate(sess, req):
    """Check against botocore's own schema - catches typos before anything is spent."""
    from botocore.validate import ParamValidator
    shape = sess.client('sagemaker').meta.service_model.operation_model('CreateTrainingJob').input_shape
    report = ParamValidator().validate(req, shape)
    if report.has_errors():
        sys.exit('INVALID REQUEST:\n' + report.generate_report())
    return True


def submit(sm, req, attempts=8):
    """A role created seconds ago is not assumable yet (IAM is eventually consistent). Retry."""
    for i in range(attempts):
        try:
            return sm.create_training_job(**req)
        except ClientError as e:
            msg = e.response['Error'].get('Message', '')
            if e.response['Error']['Code'] == 'ValidationException' and 'assume' in msg.lower() and i < attempts - 1:
                print(f'  role not assumable yet, retrying in 10s ({i + 1}/{attempts})')
                time.sleep(10)
                continue
            raise


# ---------------------------------------------------------------- job lifecycle

def logs(sess, name, tail=None):
    cw = sess.client('logs')
    group = '/aws/sagemaker/TrainingJobs'
    try:
        streams = cw.describe_log_streams(logGroupName=group, logStreamNamePrefix=f'{name}/')['logStreams']
    except ClientError:
        return []
    lines = []
    for s in streams:
        kw = {'logGroupName': group, 'logStreamName': s['logStreamName'], 'startFromHead': True}
        while True:
            r = cw.get_log_events(**kw)
            lines += [e['message'] for e in r['events']]
            if not r['events'] or r['nextForwardToken'] == kw.get('nextToken'):
                break
            kw['nextToken'] = r['nextForwardToken']
    return lines[-tail:] if tail else lines


def summary(d):
    out = {k: d.get(k) for k in ('TrainingJobStatus', 'SecondaryStatus', 'FailureReason',
                                  'TrainingTimeInSeconds', 'BillableTimeInSeconds')}
    out['InstanceType'] = d['ResourceConfig']['InstanceType']
    out['Spot'] = d.get('EnableManagedSpotTraining', False)
    out['Metrics'] = {m['MetricName']: m['Value'] for m in d.get('FinalMetricDataList', [])}
    if d.get('ModelArtifacts'):
        out['ModelArtifacts'] = d['ModelArtifacts']['S3ModelArtifacts']
    return {k: v for k, v in out.items() if v not in (None, {}, '')}


def wait(sess, name, poll=15):
    sm, last = sess.client('sagemaker'), None
    while True:
        d = sm.describe_training_job(TrainingJobName=name)
        state = f"{d['TrainingJobStatus']} / {d['SecondaryStatus']}"
        if state != last:
            print(f'  [{time.strftime("%H:%M:%S")}] {state}', flush=True)
            last = state
        if d['TrainingJobStatus'] in TERMINAL:
            break
        time.sleep(poll)
    print(json.dumps(summary(d), indent=2, default=str))
    if d['TrainingJobStatus'] != 'Completed':
        print('\n--- last 40 log lines ---')
        print('\n'.join(logs(sess, name, tail=40)) or '(no logs - failed before the container started)')
    return d


def fetch(sess, name):
    """model.tar.gz and output.tar.gz both land next to each other; extract both."""
    d = sess.client('sagemaker').describe_training_job(TrainingJobName=name)
    if d['TrainingJobStatus'] != 'Completed':
        sys.exit(f'{name} is {d["TrainingJobStatus"]}, nothing to fetch')
    model_uri = d['ModelArtifacts']['S3ModelArtifacts']
    bucket, key = model_uri[5:].split('/', 1)
    dest = ROOT / 'out' / name
    dest.mkdir(parents=True, exist_ok=True)
    s3 = sess.client('s3')
    for k in (key, key.replace('model.tar.gz', 'output.tar.gz')):
        local = dest / pathlib.Path(k).name
        try:
            s3.download_file(bucket, k, str(local))
        except ClientError:
            print(f'  (no {pathlib.Path(k).name})')
            continue
        with tarfile.open(local) as t:
            t.extractall(dest, filter='data')
        local.unlink()
    print(f'  -> {dest}: {sorted(p.name for p in dest.iterdir())}')
    return dest



def sess_ecr_token(a):
    ecr = boto3.Session(profile_name=a.profile, region_name=a.region).client('ecr')
    return ecr.get_authorization_token(registryIds=[DLC_ACCOUNT])['authorizationData'][0]


def run_local(a, req, blob):
    """Run the job in the real SageMaker container via Docker, with the same /opt/ml layout and
    config files SageMaker writes. CPU image always (the GPU one is ~10 GB). Free rehearsal."""
    import shutil, subprocess, tempfile
    ml = pathlib.Path(tempfile.mkdtemp(prefix='smlocal-'))
    cfg = ml / 'input' / 'config'
    cfg.mkdir(parents=True)
    (ml / 'source').mkdir()
    (ml / 'source' / 'sourcedir.tar.gz').write_bytes(blob)
    hp = dict(req['HyperParameters'], sagemaker_submit_directory=json.dumps('/opt/ml/source/sourcedir.tar.gz'))
    (cfg / 'hyperparameters.json').write_text(json.dumps(hp))
    (cfg / 'resourceconfig.json').write_text(json.dumps(
        {'current_host': 'algo-1', 'hosts': ['algo-1'], 'network_interface_name': 'eth0'}))
    chans = {'train': a.data, 'test': a.test_data}
    (cfg / 'inputdataconfig.json').write_text(json.dumps({c: {
        'TrainingInputMode': 'File', 'S3DistributionType': 'FullyReplicated', 'RecordWrapperType': 'None'}
        for c, d in chans.items() if d}))
    mounts = ['-v', f'{ml}:/opt/ml']
    for c, d in chans.items():
        if d:
            mounts += ['-v', f'{pathlib.Path(d).resolve()}:/opt/ml/input/data/{c}:ro']
    img = image_uri(a.region, 'ml.m5.large', a.version, a.tag)
    print(f'  local run in {img}')
    if subprocess.run(['docker', 'image', 'inspect', img], capture_output=True).returncode:
        # not pulled yet: the DLC registry needs a login, and ECR tokens expire after 12h
        import base64
        tok = sess_ecr_token(a)
        user, pw = base64.b64decode(tok['authorizationToken']).decode().split(':', 1)
        subprocess.run(['docker', 'login', '-u', user, '--password-stdin', img.split('/')[0]],
                       input=pw, text=True, check=True, capture_output=True)
        print('  pulling image (~1.7 GB, once)...', flush=True)
    # The PyTorch toolkit blocks until every host in resourceconfig resolves; SageMaker names the
    # instance algo-1, so the container must be too or it waits forever.
    r = subprocess.run(['docker', 'run', '--rm', '--hostname', 'algo-1', '--shm-size=2g', *mounts, img, 'train'])
    dest = ROOT / 'out' / a.name
    dest.mkdir(parents=True, exist_ok=True)
    for sub in ('model', 'output'):
        if (ml / sub).exists():
            shutil.copytree(ml / sub, dest / sub, dirs_exist_ok=True)
    # docker wrote these as root; hand the temp tree back before deleting it
    subprocess.run(['docker', 'run', '--rm', '-v', f'{ml}:/x', '--entrypoint', 'rm', img, '-rf', '/x/input', '/x/model',
                    '/x/output', '/x/source', '/x/checkpoints', '/x/code'], capture_output=True)
    shutil.rmtree(ml, ignore_errors=True)
    failure = dest / 'output' / 'failure'
    print(f'\n  exit {r.returncode} -> {dest}')
    if failure.exists():
        print('  FAILURE:', failure.read_text()[-1500:])
    return r.returncode

# ---------------------------------------------------------------- cli

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--profile', default=os.environ.get('AWS_PROFILE', 'amc'))
    ap.add_argument('--region', default='ap-south-1')
    ap.add_argument('--bucket', help='default: sagemaker-<account>-<region>')
    ap.add_argument('--prefix', default='amc26')
    ap.add_argument('--name')
    ap.add_argument('--data', help='local train dir to upload')
    ap.add_argument('--data-s3', help='existing s3:// train prefix instead of uploading')
    ap.add_argument('--test-data', help='local test dir to upload as the test channel')
    ap.add_argument('--test-s3', help='existing s3:// test prefix')
    ap.add_argument('--solution', help='file to ship as solution.py (default: ./solution.py)')
    ap.add_argument('--instance', default='ml.g4dn.xlarge')
    ap.add_argument('--version', default='2.9.0', help='pytorch DLC version')
    ap.add_argument('--tag', default=DEFAULT_TAG, help='pinned image tag, used if SDK lookup fails')
    ap.add_argument('--volume', type=int, default=100)
    ap.add_argument('--max-run', type=int, default=3 * 3600)
    ap.add_argument('--max-wait', type=int, default=4 * 3600)
    ap.add_argument('--spot', action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument('--hp', type=json.loads, default={}, help='hyperparameters, JSON object')
    ap.add_argument('--wait', action='store_true', help='block until the job ends')
    ap.add_argument('--local', action='store_true', help='run in the real container via docker, free')
    ap.add_argument('--create-role', action='store_true')
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--status')
    ap.add_argument('--logs')
    ap.add_argument('--fetch')
    a = ap.parse_args(argv)

    sess = boto3.Session(profile_name=a.profile, region_name=a.region)

    if a.create_role:
        return create_role(sess)
    if a.status:
        return print(json.dumps(summary(sess.client('sagemaker').describe_training_job(TrainingJobName=a.status)),
                                indent=2, default=str))
    if a.logs:
        return print('\n'.join(logs(sess, a.logs)) or '(no logs yet)')
    if a.fetch:
        return fetch(sess, a.fetch)

    acct = sess.client('sts').get_caller_identity()['Account']
    bucket = a.bucket or f'sagemaker-{acct}-{a.region}'
    val, advice = quota_advice(sess, a.instance, a.spot, a.region, a.profile)

    if a.check:
        print(f'account {acct}  region {a.region}  profile {a.profile}')
        try:
            print(f'role    {role_arn(sess)}')
        except SystemExit as e:
            print(f'role    MISSING - {e}')
        try:
            sess.client('s3').head_bucket(Bucket=bucket); print(f'bucket  s3://{bucket}')
        except ClientError:
            print(f'bucket  s3://{bucket} (created on first launch)')
        print(f'quota   {a.instance} {"spot" if a.spot else "on-demand"}: {val}')
        if advice:
            print(advice)
        return

    a.name = a.name or f'amc26-{time.strftime("%m%d-%H%M%S")}'
    sol = pathlib.Path(a.solution) if a.solution else ROOT / 'solution.py'
    problems = [] if sol.exists() else [f'no solution file at {sol}']
    if advice:
        problems.append(advice)
    if not (a.data or a.data_s3):
        problems.append('pass --data DIR or --data-s3 URI')

    if a.local:
        if not sol.exists():
            sys.exit(f'no solution file at {sol}')
        if not a.data:
            sys.exit('--local needs --data DIR (and optionally --test-data DIR)')
        blob, members = pack_source(a.solution)
        print(f'  source {members}')
        req = build_request(a, role_arn(sess, dry_run=True), 'local', 's3://local/train/', None, bucket)
        sys.exit(run_local(a, req, blob))

    if a.dry_run:
        for p in problems:
            print(f'WARN {p}', file=sys.stderr)
        src_s3 = f's3://{bucket}/{a.prefix}/source/{a.name}/sourcedir.tar.gz'
        train_s3 = a.data_s3 or f's3://{bucket}/{a.prefix}/data/{a.name}/train/'
        test_s3 = a.test_s3 or (f's3://{bucket}/{a.prefix}/data/{a.name}/test/' if a.test_data else None)
        req = build_request(a, role_arn(sess, dry_run=True), src_s3, train_s3, test_s3, bucket)
        validate(sess, req)
        print(json.dumps(req, indent=2))
        print(f'\nsource would contain: {pack_source(a.solution)[1]}')
        return print('valid. no job created.')

    if problems:
        sys.exit('cannot launch:\n  ' + '\n  '.join(problems))
    role = role_arn(sess)
    ensure_bucket(sess, bucket, a.region)
    blob, members = pack_source(a.solution)
    src_key = f'{a.prefix}/source/{a.name}/sourcedir.tar.gz'
    sess.client('s3').put_object(Bucket=bucket, Key=src_key, Body=blob)
    print(f'  source {members} -> s3://{bucket}/{src_key}')
    # Per-job prefix: a shared one keeps stale files from earlier uploads inside the next job's
    # channel. To reuse an upload without re-sending it, pass the printed URI via --data-s3/--test-s3.
    train_s3 = a.data_s3 or upload_dir(sess, a.data, bucket, f'{a.prefix}/data/{a.name}/train')
    test_s3 = a.test_s3 or (upload_dir(sess, a.test_data, bucket, f'{a.prefix}/data/{a.name}/test') if a.test_data else None)

    req = build_request(a, role, f's3://{bucket}/{src_key}', train_s3, test_s3, bucket)
    validate(sess, req)
    submit(sess.client('sagemaker'), req)
    print(f'submitted {a.name} on {a.instance}{" (spot)" if a.spot else ""}')
    if a.wait:
        return wait(sess, a.name)
    print(f'  python sm_launch.py --profile {a.profile} --status {a.name}')


if __name__ == '__main__':
    main()
