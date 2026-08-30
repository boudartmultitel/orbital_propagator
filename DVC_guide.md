# Data Versioning with DVC and Hugging Face

This project uses **DVC** to version generated datasets and a **Hugging Face Storage Bucket** as the remote storage backend.

The source code and DVC metadata are stored in Git, while potentially large generated datasets are stored separately on Hugging Face.

## Architecture

```text
GitHub
│
├── source code
├── Docker configuration
├── .dvc/
├── *.dvc
└── .gitignore
       │
       │ identifies dataset versions
       ▼
      DVC
       │
       │ push / pull
       ▼
Hugging Face Storage Bucket

Organization: TReC26-Project-2
Bucket: trajectories-data
```

The actual generated data should **not** be committed to Git.

---

## Requirements

You need:

* Docker
* Docker Compose
* access to the `TReC26-Project-2` Hugging Face organization
* access to the `trajectories-data` bucket
* your own Hugging Face S3 credentials

DVC itself is installed inside Docker, so it does not need to be installed directly on the host machine.

---

## 1. Clone the repository

```bash
git clone <repository-url>
cd orbital_propagator
```

---

## 2. Create Hugging Face credentials

Each user should use their **own Hugging Face account and token**.

Do not share credentials between users.

From Hugging Face, go to your own profile -> Settings -> Access Tokens. Create a new personal token.

Then the token menu on the right side and choose :


```text
Generate S3 credentials
```

You should receive something like:

```text
Access Key ID:
HFAKxxxxxxxxxxxxxxxx

Secret Access Key:
xxxxxxxxxxxxxxxxxxxxxxxx

```

Save the SECRET key since it will disappear after you quit the window.

---

## 3. Create `.env.dvc`

At the root of the repository, create:

```text
.env.dvc
```

with:

```env
AWS_ACCESS_KEY_ID=HFAKxxxxxxxxxxxxxxxx
AWS_SECRET_ACCESS_KEY=xxxxxxxxxxxxxxxxxxxxxxxx
```

Never commit this file.

It must be included in `.gitignore`:

```gitignore
.env.dvc
```

---

## 4. Configure your Docker UID/GID

The DVC container needs to use the same UID/GID as your host user. Otherwise, files created through Docker may belong to `root`.

Check your values:

```bash
id -u
id -g
```

Create or update the local `.env` file:

```env
HOST_UID=<your_uid>
HOST_GID=<your_gid>
```

For example:

```env
HOST_UID=10123
HOST_GID=513
```

These values are machine/user dependent.

Do not assume that all users have UID `1000`.

---

## 5. Build the DVC service

Build the image:

```bash
docker compose build dvc
```

Check that DVC is available:

```bash
docker compose run --rm dvc --version
```

---

## 6. DVC remote configuration

The repository is already configured to use the Hugging Face bucket:

```ini
[core]
    remote = hf-bucket

['remote "hf-bucket"']
    url = s3://trajectories-data/dvc-store
    endpointurl = https://s3.hf.co/TReC26-Project-2
    region = us-east-1
```

This configuration is stored in:

```text
.dvc/config
```

and is shared through Git.

Do not put credentials inside `.dvc/config`.

---

# Working with datasets

## Download the current dataset version

After cloning or pulling changes from Git:

```bash
git pull
```

retrieve the corresponding data using:

```bash
docker compose run --rm dvc pull
```

DVC reads the metadata stored in Git and downloads the correct version from Hugging Face.

---

## Add a new dataset or folder

Suppose a generated dataset is stored in:

```text
data/my_dataset/
```

Track it using:

```bash
docker compose run --rm dvc add data/my_dataset
```

DVC will typically create:

```text
data/my_dataset.dvc
```

and update a `.gitignore` file so that the real dataset is not committed to Git.

Check:

```bash
git status
```

The dataset files themselves should **not** appear as files to commit.

---

## Check dataset size before adding it to DVC

Before adding any new dataset to DVC, **always check both its total size and its number of files**. This helps prevent unexpectedly large uploads, excessive use of the shared Hugging Face storage quota, and poor DVC performance caused by very large numbers of small files.

For a dataset located at:

```text
data/my_dataset/
```

check its total size with:

```bash
du -sh data/my_dataset/
```

and count the number of files with:

```bash
find data/my_dataset/ -type f | wc -l
```

For this project, the following limits are recommended:

| Metric          |  Recommended limit | Action if exceeded                                     |
| --------------- | -----------------: | ------------------------------------------------------ |
| Dataset size    |        **≤ 10 GB** | Check whether all files are necessary before uploading |
| Number of files | **≤ 10,000 files** | Consider grouping files before using DVC               |

These are **soft limits**, not technical DVC limits. A larger dataset can still be added, but it should be reviewed by the entire team before being pushed to the shared bucket.

In particular, datasets containing tens or hundreds of thousands of small files should preferably be reorganized into larger archives or shards when possible.

Before running:

```bash
docker compose run --rm dvc add data/my_dataset
```

always perform:

```bash
du -sh data/my_dataset/
find data/my_dataset/ -type f | wc -l
```

If either recommended limit is exceeded, discuss whether the dataset should be reduced, split, or reorganized before pushing it to Hugging Face.



## Push data to Hugging Face

Upload the new or modified dataset objects using:

```bash
docker compose run --rm dvc push
```

The data will be stored in:

```text
TReC26-Project-2
└── trajectories-data
    └── dvc-store
```

DVC stores objects using hashes, so the structure visible in Hugging Face will not reproduce the original filenames directly.

This is normal.

---

## Commit the dataset version

After `dvc add` and `dvc push`, commit the DVC metadata to Git:

```bash
git add data/my_dataset.dvc data/.gitignore
git commit -m "Update dataset"
git push
```

The Git commit now identifies the exact version of the data stored remotely.

---

# Typical workflow

## Before starting work

Update the source code:

```bash
git pull --rebase
```

Then retrieve the associated datasets:

```bash
docker compose run --rm dvc pull
```

---

## Generate new data

Create and execute a versioned trajectory manifest using the
`orbital_propagator` service. For example:

```bash
docker compose run --rm orbital_propagator manifest append \
  --manifest /shared/data/manifests/trajectories.jsonl \
  --recipe multi_planet_two_body --count 100 --seed 42

docker compose run --rm orbital_propagator manifest build \
  --manifest /shared/data/manifests/trajectories.jsonl \
  --output-dir /shared/data/datasets/trajectories
```

Generated data should be written into the shared project data directories.

---

## Version the generated data

For example:

```bash
docker compose run --rm dvc add data/trajectories
```

Then upload it:

```bash
docker compose run --rm dvc push
```

Then commit the metadata:

```bash
git add data/trajectories.dvc data/.gitignore
git commit -m "Generate new trajectory dataset"
git push
```

---

# Retrieving an older dataset version

Because DVC metadata is versioned with Git, an older Git commit can also restore the corresponding dataset.

For example:

```bash
git checkout <commit>
```

Then:

```bash
docker compose run --rm dvc pull
```

DVC will retrieve the data corresponding to that Git revision.

Return to the latest version with:

```bash
git checkout main
git pull
docker compose run --rm dvc pull
```

---

# Important rules

## Never commit generated data directly to Git

Do not use:

```bash
git add data/my_dataset/
```

for directories managed by DVC.

Instead use:

```bash
docker compose run --rm dvc add data/my_dataset
```

and commit only:

```text
*.dvc
.gitignore
```

---

## Never commit credentials

Never commit:

```text
.env.dvc
```

and never add credentials directly to:

```text
docker-compose.yml
Dockerfile
.dvc/config
```

---

## Each user should have their own credentials

Do not share one Hugging Face token or S3 secret between the whole team.

Each user should have:

```text
Hugging Face account
        ↓
personal token
        ↓
personal S3 credentials
        ↓
.env.dvc
```

---

## Do not use `docker compose up` for DVC

DVC is configured as a tool service rather than a persistent application.

Use:

```bash
docker compose run --rm dvc <command>
```

Examples:

```bash
docker compose run --rm dvc status

docker compose run --rm dvc add data/trajectories

docker compose run --rm dvc push

docker compose run --rm dvc pull
```

---

# Checking DVC status

To check whether local data and DVC metadata are synchronized:

```bash
docker compose run --rm dvc status
```

To check Git separately:

```bash
git status
```

Both commands serve different purposes:

```text
git status
    → source code and DVC metadata

dvc status
    → datasets and DVC pipeline state
```

---

# Current remote

```text
Hugging Face organization:
TReC26-Project-2

Storage Bucket:
trajectories-data

DVC remote:
s3://trajectories-data/dvc-store

S3 endpoint:
https://s3.hf.co/TReC26-Project-2
```
