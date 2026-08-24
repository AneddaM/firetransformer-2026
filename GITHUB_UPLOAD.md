# Uploading this repository to GitHub

## Recommended: GitHub CLI

After extracting the ZIP, open a terminal inside the repository root:

```bash
git init
git branch -M main
git add .
git commit -m "WF-IoT camera-ready reproducibility revision"
gh repo create firetransformer-wfiot-2026 --public --source=. --remote=origin --push
```

If the target repository already exists:

```bash
git init
git branch -M main
git add .
git commit -m "WF-IoT camera-ready reproducibility revision"
git remote add origin https://github.com/<ACCOUNT>/<REPOSITORY>.git
git push -u origin main
```

## GitHub website

Create an empty repository, extract this ZIP locally, then upload the extracted files
and directories. Do not upload the ZIP itself expecting GitHub to unpack it.

## Before making the repository public

- decide whether to add a software license;
- verify author names / repository URL in `CITATION.cff`;
- do not commit raw dataset files unless permitted and desired;
- do not commit provisional or estimated experimental metrics;
- run `python tests/smoke_test.py`.
