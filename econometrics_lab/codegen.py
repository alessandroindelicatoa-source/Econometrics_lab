def generate_code(language, family, y, x, extra=None):
    rhs=" + ".join(x) if x else "1"
    if language=="Python":
        if family=="OLS": return f'import statsmodels.formula.api as smf\nmodel = smf.ols("{y} ~ {rhs}", data=df).fit(cov_type="HC3")\nprint(model.summary())'
        if family=="Logit": return f'import statsmodels.formula.api as smf\nmodel = smf.logit("{y} ~ {rhs}", data=df).fit()\nprint(model.summary())\nprint(model.get_margeff(at="overall").summary())'
        if family=="Probit": return f'import statsmodels.formula.api as smf\nmodel = smf.probit("{y} ~ {rhs}", data=df).fit()\nprint(model.summary())'
        if family=="Poisson": return f'import statsmodels.api as sm\nimport statsmodels.formula.api as smf\nmodel = smf.glm("{y} ~ {rhs}", data=df, family=sm.families.Poisson()).fit()\nprint(model.summary())'
    if language=="R":
        if family=="OLS": return f'm <- lm({y} ~ {rhs}, data=df)\nsummary(m)\n# Robust SE: lmtest::coeftest(m, vcov=sandwich::vcovHC(m, type="HC3"))'
        if family=="Logit": return f'm <- glm({y} ~ {rhs}, data=df, family=binomial(link="logit"))\nsummary(m)\n# margins::margins(m)'
        if family=="Probit": return f'm <- glm({y} ~ {rhs}, data=df, family=binomial(link="probit"))\nsummary(m)'
        if family=="Poisson": return f'm <- glm({y} ~ {rhs}, data=df, family=poisson(link="log"))\nsummary(m)'
    if language=="Stata":
        cmd={"OLS":"reg","Logit":"logit","Probit":"probit","Poisson":"poisson"}.get(family,"reg")
        return f'{cmd} {y} {" ".join(x)}, vce(robust)'
    if language=="Gretl":
        if family=="OLS": return f'ols {y} const {" ".join(x)} --robust'
        if family=="Logit": return f'logit {y} const {" ".join(x)}'
        if family=="Probit": return f'probit {y} const {" ".join(x)}'
        if family=="Poisson": return f'poisson {y} const {" ".join(x)}'
    return "# Code template not yet available for this model."
