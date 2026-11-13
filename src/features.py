import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
class BankFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        z=X.copy()
        z["was_previously_contacted"]=(z["pdays"]!=999).astype(int)
        z["days_since_previous_contact"]=z["pdays"].where(z["pdays"]!=999,0)
        z=z.drop(columns=["pdays"])
        z["total_campaign_contacts"]=z["campaign"]+z["previous"]
        z["age_group"]=pd.cut(z["age"],bins=[0,25,35,45,55,120],labels=["18-25","26-35","36-45","46-55","56+"]).astype("object")
        return z
