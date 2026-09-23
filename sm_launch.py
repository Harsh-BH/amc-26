"""Submit a SageMaker training job with boto3. No sagemaker SDK - it drags torch+CUDA in
and that is what filled this disk once already.

    python sm_launch.py --create-role                      # once, day 1
    python sm_launch.py --data data/ --dry-run             # check the payload, spend nothing
    python sm_launch.py --data data/ --instance ml.g4dn.xlarge --spot
    python sm_launch.py --status <job-name>
    python sm_launch.py --fetch  <job-name>                # download model + output

Spot is on by default: managed spot training is billed at the spot price, typically
~70% off, and SageMaker handles the interruption/restart. Checkpoint in solution.py if
your job runs longer than ~1h, otherwise an interruption restarts from zero.
"""
import argparse, io, json, os, pathlib, sys, tarfile, time
import boto3
from botocore.exceptions import ClientError

# DLC PyTorch training image. Account 763104351884 serves ap-south-1 (verified against
# the sagemaker-core image_uri_config). The short tag tracks patch releases.
DLC_ACCOUNT = '763104351884'
DEFAULT_TAG = '2.9-gpu-py312'
ROLE_NAME = 'AMCSageMakerExecutionRole'
SRC_FILES = ('sm_train.py', 'solution.py', 'metrics.py', 'baseline.py', 'requirements.txt')


def image_uri(region, tag, cpu=False):
    t = tag.replace('-gpu-', '-cpu-') if cpu else tag
    return f'{DLC_ACCOUNT}.dkr.ecr.{region}.amazonaws.com/pytorch-training:{t}'


def create_role(sess):
    """SageMaker needs a role it can assume. This is the day-1 blocker; do it once."""
    iam = sess.client('iam')
    trust = {'Version': '2012-10-17', 'Statement': [{'Effect': 'Allow',
             'Principal': {'Service': 'sagemaker.amazonaws.com'}, 'Action': 'sts:AssumeRole'}]}
    try:
        arn = iam.create_role(RoleName=ROLE_NAME, AssumeRolePolicyDocument=json.dumps(trust),
                              Description='Amazon ML Challenge training jobs')['Role']['Arn']
        print(f'created {arn}')
    except iam.exceptions.EntityAlreadyExistsException:
        arn = iam.get_role(RoleName=ROLE_NAME)['Role']['Arn']
        print(f'exists {arn}')
    for p in ('arn:aws:iam::aws:policy/AmazonSageMakerFullAccess',
              'arn:aws:iam::aws:policy/AmazonS3FullAccess'):
        iam.attach_role_policy(RoleName=ROLE_NAME, PolicyArn=p)
    print('attached SageMakerFullAccess + S3FullAccess')
    return arn


def role_arn(sess, given, dry_run=False):
    if given:
        return given
    iam = sess.client('iam')
    try:
        return iam.get_role(RoleName=ROLE_NAME)['Role']['Arn']
    except ClientError:
        if dry_run:   # a dry run only shapes the payload; it never needs a usable role
            return f'arn:aws:iam::000000000000:role/{ROLE_NAME}'
        sys.exit(f'no role {ROLE_NAME}. Run: python sm_launch.py --create-role')


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
    return bucket


def upload_source(sess, bucket, prefix):
    """Tar the scripts SageMaker should run. This is what 'script mode' actually is."""
    root = pathlib.Path(__file__).parent
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w:gz') as tar:
        for name in SRC_FILES:
            p = root / name
            if p.exists():
                tar.add(p, arcname=name)
    buf.seek(0)
    key = f'{prefix}/source/sourcedir.tar.gz'
    sess.client('s3').put_object(Bucket=bucket, Key=key, Body=buf.getvalue())
    print(f'  source -> s3://{bucket}/{key}')
    return f's3://{bucket}/{key}'


def upload_data(sess, local, bucket, prefix):
    s3, n = sess.client('s3'), 0
    local = pathlib.Path(local)
    for f in sorted(local.rglob('*')):
        if f.is_file() and not f.name.startswith('.'):
            s3.upload_file(str(f), bucket, f'{prefix}/train/{f.relative_to(local)}')
            n += 1
    print(f'  data -> s3://{bucket}/{prefix}/train/ ({n} files)')
    return f's3://{bucket}/{prefix}/train/'


def build_request(a, region, role, src_s3, data_s3, bucket, prefix):
    hp = {'sagemaker_program': 'sm_train.py', 'sagemaker_submit_directory': src_s3}
    hp.update({k: json.dumps(v) if not isinstance(v, str) else v for k, v in a.hp.items()})
    req = {
        'TrainingJobName': a.name,
        'RoleArn': role,
        'AlgorithmSpecification': {
            'TrainingImage': image_uri(region, a.tag, cpu=a.instance.startswith(('ml.m', 'ml.c', 'ml.t'))),
            'TrainingInputMode': 'File',
        },
        'HyperParameters': hp,
        'InputDataConfig': [{
            'ChannelName': 'train',
            'DataSource': {'S3DataSource': {
                'S3DataType': 'S3Prefix', 'S3Uri': data_s3, 'S3DataDistributionType': 'FullyReplicated'}},
        }],
        'OutputDataConfig': {'S3OutputPath': f's3://{bucket}/{prefix}/output/'},
        'ResourceConfig': {'InstanceType': a.instance, 'InstanceCount': 1, 'VolumeSizeInGB': a.volume},
        'StoppingCondition': {'MaxRuntimeInSeconds': a.max_run},
    }
    if a.spot:
        req['EnableManagedSpotTraining'] = True
        # MaxWait must be >= MaxRuntime; it caps runtime + time queued waiting for capacity.
        req['StoppingCondition']['MaxWaitTimeInSeconds'] = max(a.max_wait, a.max_run)
        req['CheckpointConfig'] = {'S3Uri': f's3://{bucket}/{prefix}/checkpoints/'}
    return req


def validate(sess, req):
    """Check the payload against botocore's own schema. Catches typos without spending money."""
    from botocore.validate import ParamValidator
    model = sess.client('sagemaker').meta.service_model.operation_model('CreateTrainingJob')
    report = ParamValidator().validate(req, model.input_shape)
    if report.has_errors():
        sys.exit('INVALID REQUEST:\n' + report.generate_report())
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--profile', default=os.environ.get('AWS_PROFILE', 'amc'))
    ap.add_argument('--region', default='ap-south-1')
    ap.add_argument('--bucket', help='default: sagemaker-<account>-<region>')
    ap.add_argument('--prefix', default='amc26')
    ap.add_argument('--name', default=None)
    ap.add_argument('--data', help='local dir to upload as the train channel')
    ap.add_argument('--data-s3', help='use an existing s3:// prefix instead of uploading')
    ap.add_argument('--instance', default='ml.g4dn.xlarge')
    ap.add_argument('--tag', default=DEFAULT_TAG)
    ap.add_argument('--volume', type=int, default=100)
    ap.add_argument('--max-run', type=int, default=3 * 3600)
    ap.add_argument('--max-wait', type=int, default=4 * 3600)
    ap.add_argument('--spot', action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument('--hp', type=json.loads, default={}, help='hyperparameters as a JSON object')
    ap.add_argument('--create-role', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--status')
    ap.add_argument('--fetch')
    a = ap.parse_args()

    sess = boto3.Session(profile_name=a.profile, region_name=a.region)
    sm = sess.client('sagemaker')

    if a.create_role:
        create_role(sess); return
    if a.status:
        d = sm.describe_training_job(TrainingJobName=a.status)
        print(json.dumps({k: str(d.get(k)) for k in
              ('TrainingJobStatus', 'SecondaryStatus', 'FailureReason', 'TrainingTimeInSeconds',
               'BillableTimeInSeconds', 'ModelArtifacts')}, indent=2)); return
    if a.fetch:
        d = sm.describe_training_job(TrainingJobName=a.fetch)
        uri = d['ModelArtifacts']['S3ModelArtifacts']
        b, k = uri[5:].split('/', 1)
        pathlib.Path('out').mkdir(exist_ok=True)
        sess.client('s3').download_file(b, k, 'out/model.tar.gz')
        print('out/model.tar.gz'); return

    acct = sess.client('sts').get_caller_identity()['Account']
    bucket = a.bucket or f'sagemaker-{acct}-{a.region}'
    a.name = a.name or f'amc26-{int(time.time())}'
    role = role_arn(sess, None, a.dry_run)

    if a.dry_run:
        src_s3, data_s3 = f's3://{bucket}/{a.prefix}/source/sourcedir.tar.gz', a.data_s3 or f's3://{bucket}/{a.prefix}/train/'
    else:
        ensure_bucket(sess, bucket, a.region)
        src_s3 = upload_source(sess, bucket, a.prefix)
        data_s3 = a.data_s3 or (upload_data(sess, a.data, bucket, a.prefix) if a.data
                                else sys.exit('pass --data or --data-s3'))

    req = build_request(a, a.region, role, src_s3, data_s3, bucket, a.prefix)
    validate(sess, req)
    if a.dry_run:
        print(json.dumps(req, indent=2)); print('\nvalid. no job created.'); return

    sm.create_training_job(**req)
    print(f'\nsubmitted {a.name} on {a.instance}{" (spot)" if a.spot else ""}')
    print(f'  watch: python sm_launch.py --status {a.name}')
    print(f'  logs:  aws logs tail /aws/sagemaker/TrainingJobs --log-stream-name-prefix {a.name} --follow --profile {a.profile} --region {a.region}')


if __name__ == '__main__':
    main()
