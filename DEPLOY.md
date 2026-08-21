# Deploy Econometrics Lab V2

1. Create or open your GitHub repository.
2. Replace the previous project files with the contents of this V2 folder.
3. Keep the directory structure exactly as provided.
4. Commit the changes.
5. Streamlit Community Cloud should redeploy automatically.

If creating a new app:
- Repository: your Econometrics Lab repository
- Branch: `main`
- Main file: `app.py`

Recommended Python version: 3.12.

## First smoke test

Load **Data → Panel / causal research demo**.

Then:

### OLS
Model Studio → Cross-sectional
- Y: `wage`
- X: `education`, `age`, `female`, `immigrant`
- SE: `HC3`

### Logit
- Y: `employed`
- X: `education`, `age`, `female`, `immigrant`

### DiD
Model Studio → Causal → Difference-in-Differences
- Y: `outcome`
- Treatment: `treat`
- Post: `post`
- Unit: `id`
- Time: `year`

### Research Lab
Use `outcome` as Y and `treat` or another focal predictor to test the robustness interface.
