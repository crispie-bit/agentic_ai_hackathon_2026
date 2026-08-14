# AWS setup — do this BEFORE Day 2

Twenty minutes, once. Session 1 needs none of it; Session 2 needs all of it
working before the session starts, because thirty people doing this
simultaneously on venue wifi does not fit in a coffee break.

You need four things by the end:

1. the AWS CLI installed
2. an SSO profile configured
3. an active SSO login
4. Bedrock model access granted in your region

`uv run 00_check_bedrock.py` tests 2, 3 and 4 and tells you which one broke.

---

## 1. Install the AWS CLI (v2)

**macOS**

```bash
brew install awscli
# or, without Homebrew:
curl "https://awscli.amazonaws.com/AWSCLIV2.pkg" -o AWSCLIV2.pkg
sudo installer -pkg AWSCLIV2.pkg -target /
```

**Windows** — download and run the MSI:
<https://awscli.amazonaws.com/AWSCLIV2.msi>
Then **close and reopen your terminal**, or `aws` will not be on PATH.

**Linux**

```bash
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o awscliv2.zip
unzip awscliv2.zip && sudo ./aws/install
```

Check it:

```bash
aws --version
# aws-cli/2.x.x Python/3.x.x ...
```

**It must say 2.x.** SSO does not work properly on v1, and some systems have an
old v1 lurking from a `pip install awscli` years ago. If you see `1.x`, remove
it before continuing.

---

## 2. Get these four values from whoever runs your AWS account

You cannot invent them, and you cannot proceed without them:

| Value | Looks like |
|---|---|
| SSO start URL | `https://d-xxxxxxxxxx.awsapps.com/start` |
| SSO region | `ap-southeast-1` |
| Account ID | a 12-digit number |
| Role name | `AdministratorAccess`, `PowerUserAccess`, … |

If you are using a personal AWS account with no SSO, skip to
**"No SSO?"** at the bottom.

---

## 3. Configure the profile

```bash
aws configure sso
```

It asks, in this order:

```
SSO session name (Recommended): workshop
SSO start URL [None]:           https://d-xxxxxxxxxx.awsapps.com/start
SSO region [None]:              ap-southeast-1
SSO registration scopes [sso:account:access]:   <press Enter>
```

A browser opens. Sign in and **Allow** the request. Back in the terminal:

```
The only AWS account available to you is: 123456789012   <- or pick from a list
Using the role name "AdministratorAccess"                <- or pick from a list
CLI default client Region [None]: ap-southeast-1
CLI default output format [None]: json
CLI profile name [...]:           workshop
```

Two of those matter more than they look:

- **CLI default client Region** is where your Bedrock calls go. Set it to
  `ap-southeast-1`. It is a separate setting from the SSO region, and leaving it
  blank is a common cause of "it worked for them, not for me".
- **CLI profile name** is what you type in every later command. Keep it short.

This writes `~/.aws/config`. No secrets are stored — that is the point of SSO.

---

## 4. Log in

```bash
aws sso login --profile workshop
```

Browser opens, you approve, done. Verify:

```bash
aws sts get-caller-identity --profile workshop
```

Expect something like:

```json
{
  "Account": "123456789012",
  "Arn": "arn:aws:sts::123456789012:assumed-role/AWSReservedSSO_AdministratorAccess_.../you@example.com"
}
```

**Sessions expire, typically after 8–12 hours.** You will run
`aws sso login` again on the morning of Day 2. That is normal, and it is the
security benefit: a leaked temporary credential is worthless tomorrow.

---

## 5. Enable Bedrock model access

**Working credentials are not the same as a working call.** Model access is
granted per model *and* per region, and it is off by default.

1. AWS console → **Bedrock** → make sure the region selector says
   **ap-southeast-1**
2. Left nav → **Model access**
3. **Enable specific models** → tick **Anthropic Claude Haiku 4.5**
4. Submit. It is usually instant.

If you do not have permission to do this, the account admin must — it cannot be
done from code.

---

## 6. Point the lab at it

In `lab/.env`:

```
LLM_PROVIDER=bedrock
AWS_PROFILE=workshop
AWS_DEFAULT_REGION=ap-southeast-1
```

`_common.py` loads that file, and boto3 reads `AWS_PROFILE` from the
environment. No keys anywhere.

Then:

```bash
cd agentic_teaching/lab
uv sync --extra aws
uv run 00_check_bedrock.py
```

Expected:

```
  caller arn:aws:sts::123456789012:assumed-role/...
  invoking the model...
  model said: 'ready'
  checking langchain-aws...
  langchain said: 'ready'

OK — you are ready for every lab in this session
```

To send the Session 1 labs back to Groq, comment out `LLM_PROVIDER=bedrock`.
That one line is the entire difference between the two providers.

---

## Troubleshooting

| What you see | What it means | Fix |
|---|---|---|
| `command not found: aws` | not installed, or PATH not reloaded | reopen the terminal |
| `Token has expired and refresh failed` | SSO session ran out | `aws sso login --profile workshop` |
| `The config profile (workshop) could not be found` | typo, or `aws configure sso` never finished | `aws configure list-profiles` |
| `AccessDeniedException` on invoke | credentials fine, **model access not granted** | step 5, and check the region |
| `ValidationException` naming the model id | model not available in that region | `export BEDROCK_MODEL=<id from the console>` |
| `UnrecognizedClientException` | stale or partial credentials in env vars | `unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY` |
| Works in the CLI, fails in Python | different region or profile | check `AWS_PROFILE` / `AWS_DEFAULT_REGION` in `.env` |

Useful commands:

```bash
aws configure list-profiles          # which profiles exist
aws configure list --profile workshop # what a profile resolves to
aws sso logout                        # end the session early
```

---

## No SSO?

If you are on a personal account, use an IAM user with access keys instead.
Less safe, so treat the keys accordingly.

1. Console → **IAM** → Users → your user → **Security credentials**
2. **Create access key** → *Command Line Interface* → copy both values
3. Either `aws configure` (writes `~/.aws/credentials`), or put them in
   `lab/.env`:

```
LLM_PROVIDER=bedrock
AWS_DEFAULT_REGION=ap-southeast-1
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...
```

If the keys are **temporary** (from STS, or a workshop/sandbox account) there
is a third value and it is mandatory:

```
AWS_SESSION_TOKEN=...
```

Rules, briefly: `.env` is gitignored — keep it that way. Never paste a key into
chat, a screenshot or a shared doc. Delete the key in the console when the
workshop ends; that takes ten seconds and costs nothing.

---

## Cost

Everything in Session 2 is metered per token. The whole lab is cents, not
dollars — but an agent loop left running is not. Before Day 2:

- AWS console → **Billing** → **Budgets** → create a small monthly budget with
  an email alert
- Every lab file prints its own token usage; `section_3_bedrock/07_tokens_and_cost.py`
  turns that into a dollar estimate

Set the budget before the hackathon, not after.
