# GitHub + Streamlit Cloud deployment

## 1. Create the repository
Create a new GitHub repository, for example:

`econometrics-lab`

## 2. Upload the project
Upload the CONTENTS of the `econometrics_lab` project folder to the root of the repository.

The GitHub root should look like:

```text
app.py
requirements.txt
README.md
DEPLOY.md
.streamlit/
data/
econometrics_lab/
```

Do not upload an extra outer folder if GitHub would make the path `econometrics_lab/econometrics_lab/app.py`.

## 3. Deploy
In Streamlit Community Cloud:

1. New app
2. Select your GitHub repository
3. Branch: `main`
4. Main file path: `app.py`
5. Deploy

If the platform asks for a Python version, use Python 3.12.

## 4. First test
Open:

**Data → Import → Built-in demos → Panel / causal demo → Load demo**

Then try:

### OLS
- Dependent: `wage`
- Regressors: `education`, `age`, `female`, `immigrant`
- Covariance: `HC3`

### Logit
- Dependent: `employed`
- Regressors: `education`, `age`, `female`, `immigrant`

### Panel FE
- Entity: `id`
- Time: `year`
- Dependent: `outcome`
- Regressors: `education`, `age`, `treat`, `post`
- Time fixed effects: ON
- Covariance: clustered

### DiD
- Outcome: `outcome`
- Treatment: `treat`
- Post: `post`
- Unit: `id`
- Time: `year`

The synthetic data were constructed so that the DiD interaction is positive and detectable.

## 5. Graphs
Use **Graphs** as a standalone visual workspace or estimate models first and then use **Graphs → Model graphs**.

## 6. No secrets required
The current version does not require API keys, passwords or external AI services.
