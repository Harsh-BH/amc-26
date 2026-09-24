# Claiming the AWS credits

How to get the full **$200** on a new AWS account, or the rest of it on one you already
have. This worked on our account on 24 Sep 2026: $45.31 → $125.31, and the whole run cost about one cent.

## What you get

| | Amount | How |
|---|---|---|
| Sign-up | **$100** | Automatic when a *new* account is created |
| 5 activities | **5 × $20** | Do each one once (listed below) |
| **Total** | **$200** | |

- **New account (for the hackathon):** sign up at `aws.amazon.com/free` and choose the
  **Free plan**, as the challenge's setup guide says. Then do all 5 activities.
- **Account created after July 2025:** you already got the $100. Check which activities are
  still open and claim those.
- **New customers only.** AWS rules out anyone who *"has an existing AWS account or has had one in the
  past"*. Don't open a second account for yourself; AWS can link them by phone number or card.

## Before you start

1. Switch the region (top-right of the console) to **Asia Pacific (Mumbai) `ap-south-1`**.
2. Check where you stand:
   - **Console Home → "Explore AWS" widget** lists the 5 activities, each marked open or done.
   - **Billing and Cost Management → Free Tier** shows remaining credits and plan expiry.

   Or from a terminal:

   ```bash
   aws freetier get-account-plan-state --region us-east-1
   aws freetier list-account-activities --region us-east-1 \
     --query 'activities[].[status,reward.credit.amount,title]' --output text
   ```

## What we learned doing it

- **No widget clicks needed.** AWS detects the actions themselves, whether you use the console or the CLI.
- **It registers 10–15 minutes after you finish.** Credits post a minute or two after that.
- **Some activities count on create, some on clean-up.** RDS completed as soon as the
  database became *Available*. EC2 completed when the instance was *terminated*. Either
  way: create it, use it, delete it.

## The five activities — $20 each

**Start with RDS.** It takes ~10 minutes to come up, so do the others while you wait.

### Create an Aurora or RDS database

[Open the guided tutorial](https://console.aws.amazon.com/rds/home#launch-dbinstance:tutorialId=create-rds-aurora-database)

1. RDS → **Create database** → PostgreSQL → template **Free tier / Sandbox**.
2. Instance class **db.t4g.micro**, 20 GB storage, **Public access: No**.
3. Create, and wait until status is **Available** (~10 min).
4. **Delete it right away:** Actions → Delete → untick *Create final snapshot*, untick
   *Retain automated backups* → type `delete me`.

> ⚠️ A forgotten database eats credits fast. db.t4g.micro is $0.021/hr, so ~$15/month
> if you leave it running. Delete it the moment it's Available.

<details><summary>CLI</summary>

```bash
aws rds create-db-instance --db-instance-identifier credit-activity-db --engine postgres \
  --db-instance-class db.t4g.micro --allocated-storage 20 --storage-type gp2 \
  --master-username dbadmin --master-user-password '<any 16+ char password>' \
  --no-publicly-accessible --no-multi-az --backup-retention-period 0 --region ap-south-1
aws rds wait db-instance-available --db-instance-identifier credit-activity-db --region ap-south-1
aws rds delete-db-instance --db-instance-identifier credit-activity-db \
  --skip-final-snapshot --delete-automated-backups --region ap-south-1
```
</details>

### Use a foundation model in the Amazon Bedrock playground

[Open the guided tutorial](https://console.aws.amazon.com/bedrock/home#text-generation-playground?tutorialId=playground-get-started)

1. Bedrock → **Playgrounds → Chat**.
2. Select a model. **Amazon Nova Micro** is the cheapest.
3. Type any prompt → **Run**. One prompt costs a fraction of a cent.

Nothing to clean up. This one took the longest to register for us (~15 min).

<details><summary>CLI</summary>

```bash
aws bedrock-runtime converse --model-id apac.amazon.nova-micro-v1:0 --region ap-south-1 \
  --messages '[{"role":"user","content":[{"text":"Say hi"}]}]' --inference-config '{"maxTokens":20}'
```
</details>

### Set up a cost budget using AWS Budgets

[Open the guided tutorial](https://console.aws.amazon.com/costmanagement/home#/budgets/create?tutorialId=create-cost-budget-simplified)

1. Billing and Cost Management → **Budgets → Create budget**.
2. Monthly cost budget, e.g. **$40**, with **your email** for alerts.
3. Create. **Keep this one:** it's free and it's your early warning.

> 💡 By default a budget counts spend *after* credits, which is ~$0 while credits last, so
> the alert never fires. To track credit burn, use *Customize* and exclude **Credits** from
> the charge types.

<details><summary>CLI</summary>

```bash
aws budgets create-budget --account-id <your-12-digit-account-id> --region us-east-1 --budget '{
  "BudgetName": "credit-burn", "BudgetLimit": {"Amount": "40", "Unit": "USD"},
  "TimeUnit": "MONTHLY", "BudgetType": "COST",
  "CostTypes": {"IncludeCredit": false, "IncludeRefund": false}}'
```
The CLI version has no email alert; add one in the console afterwards.
</details>

### Create a web app using AWS Lambda

[Open the guided tutorial](https://console.aws.amazon.com/lambda/home#/tutorial/furl)

1. Lambda → **Create function** → Author from scratch → runtime **Python 3.12** → Create.
2. Configuration → **Function URL → Create function URL**. Auth type **AWS_IAM** keeps it
   private. If the tutorial has you use **NONE**, delete it straight after.
3. Open the URL to check it responds.
4. Clean up: **Actions → Delete** the function, then delete the role Lambda created for it
   in IAM (named `<function-name>-role-…`).

Effectively free.

### Launch an instance using EC2

[Open the guided tutorial](https://console.aws.amazon.com/ec2/home#LaunchInstances:tutorialId=launch-a-test-instance)

1. EC2 → **Launch instance** → Amazon Linux, type **t3.micro**.
2. Launch, wait until it's *Running*.
3. **Terminate it** (Instance state → Terminate). The activity completes on termination.

A few minutes of t3.micro costs well under a cent. *(We claimed this one earlier through a
normal instance launch rather than these exact steps.)*

## Check it worked

```bash
aws freetier list-account-activities --region us-east-1 \
  --query 'activities[].[status,title]' --output text     # all five: COMPLETED
aws freetier get-account-plan-state --region us-east-1 \
  --query accountPlanRemainingCredits.amount               # should be $80–100 higher than before
```

Then make sure nothing is left running: no RDS databases, no RDS snapshots, no EC2
instances, no Lambda functions. The budget is the only thing to keep.

## Get GPU quota — do this tonight

New AWS accounts can't run *any* SageMaker training job until AWS raises a limit (a
*quota*). The request is **free**: a quota is only permission, not compute. You pay per hour, out of
your credits, only while a job runs. A person at AWS reviews it. On our account it took
**under 20 hours**, so file it now and use **Kaggle** GPUs until it lands.

**Console:** region **Asia Pacific (Mumbai)** → **Service Quotas** → **Amazon SageMaker**
→ search each of these → **Request increase at account level** → value **1**:

- `ml.g4dn.xlarge for training job usage`
- `ml.g4dn.xlarge for spot training job usage`

**Or CLI:**

```bash
aws service-quotas request-service-quota-increase --service-code sagemaker \
  --quota-code L-3F53BF0F --desired-value 1 --region ap-south-1   # g4dn.xlarge training
aws service-quotas request-service-quota-increase --service-code sagemaker \
  --quota-code L-944F78BB --desired-value 1 --region ap-south-1   # g4dn.xlarge spot training
```

If it says the value must be *greater than the default*, use the default + 1. Check
progress under **Service Quotas → Quota request history**. A case can show *closed* before
the new value appears.

**What happened on ours:** we asked for CPU quota only (`ml.m5.large`). AWS then raised the
whole account: most CPU types to 15, plus **1 T4 (`g4dn.xlarge`) and 1 A10G (`g5.xlarge`)**.
Ask for the GPU directly anyway; a newer account may not get the same treatment.

**Why everyone should do it:** each account gets **1 GPU at a time**. Four approved
accounts means **four GPU jobs in parallel**, each paid from its owner's credits.

| GPU | On-demand | Spot (our test: ~77% off) | 72 h on one GPU |
|---|---|---|---|
| T4 `ml.g4dn.xlarge` | $0.81/hr | ~$0.19/hr | ~$13 spot · ~$58 on-demand |

Your credit balance updates hours after you spend, so watch job hours, not the balance.

## Rules that will bite you

- **On the Free plan, the account closes when credits hit $0**, and you lose access to
  its S3 data and models. Keep code in git and copy results to your laptop as you go.
- **SageMaker's "250 hrs notebook + 50 hrs training"** is a short-term trial. AWS
  only turns those trials on for **Paid-plan** accounts.
- **The top-500 bonus ($100 at the 48h mark)** is most likely a promotional credit, and
  Free-plan accounts *"are not eligible for other promotional credits"*. You'd probably need to
  upgrade to Paid to redeem it. Upgrading keeps your remaining credits, but anything
  beyond them is then charged to your card.
- **SageMaker training quota starts at 0 on new accounts**, GPUs included. See
  [Get GPU quota](#get-gpu-quota--do-this-tonight) above.

Sources: [AWS: Choosing a plan](https://docs.aws.amazon.com/awsaccountbilling/latest/aboutv2/free-tier-plans.html) ·
[AWS: Earning additional credits](https://docs.aws.amazon.com/awsaccountbilling/latest/aboutv2/free-tier-plans-activities.html) ·
[AWS Free Tier FAQs](https://aws.amazon.com/free/free-tier-faqs/) ·
[Challenge setup guide (PDF)](https://d8it4huxumps7.cloudfront.net/files/6a9e7a9ddd441_guide_aws_builder_account.pdf)
