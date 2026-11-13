import json
from pathlib import Path
import pandas as pd
import streamlit as st
from joblib import load
ROOT=Path(__file__).parent
st.set_page_config(page_title="Bank Marketing Intelligence",page_icon="🏦",layout="wide")
st.title("🏦 Bank Marketing Intelligence")
st.subheader("Predicting customer subscription to a term deposit")
st.caption("Decision-support estimate only; predictions are not guarantees.")
model=load(ROOT/"artifacts/best_model.joblib")
meta=json.loads((ROOT/"artifacts/model_metadata.json").read_text())
threshold=float(meta["decision_threshold"])
with st.sidebar:
 st.header("Customer profile")
 age=st.slider("Age",18,95,35)
 job=st.selectbox("Job",["admin.","blue-collar","entrepreneur","housemaid","management","retired","self-employed","services","student","technician","unemployed","unknown"])
 marital=st.selectbox("Marital status",["married","single","divorced","unknown"])
 education=st.selectbox("Education",["university.degree","high.school","basic.9y","basic.6y","basic.4y","professional.course","illiterate","unknown"])
 default=st.selectbox("Credit in default",["no","yes","unknown"])
 housing=st.selectbox("Housing loan",["no","yes","unknown"])
 loan=st.selectbox("Personal loan",["no","yes","unknown"])
 contact=st.selectbox("Contact type",["cellular","telephone"])
 month=st.selectbox("Last contact month",["mar","apr","may","jun","jul","aug","sep","oct","nov","dec"])
 day=st.selectbox("Day of week",["mon","tue","wed","thu","fri"])
 campaign=st.number_input("Contacts in this campaign",1,60,1)
 pdays=st.number_input("Days since previous contact (999 = never)",0,999,999)
 previous=st.number_input("Previous campaign contacts",0,20,0)
 poutcome=st.selectbox("Previous outcome",["nonexistent","failure","success"])
 emp=st.number_input("Employment variation rate",value=1.1)
 cpi=st.number_input("Consumer price index",value=93.994)
 cci=st.number_input("Consumer confidence index",value=-36.4)
 euribor=st.number_input("Euribor 3-month rate",value=4.857)
 employed=st.number_input("Number of employees",value=5191.0)
row=pd.DataFrame([{"age":age,"job":job,"marital":marital,"education":education,"default":default,"housing":housing,
 "loan":loan,"contact":contact,"month":month,"day_of_week":day,"campaign":campaign,"pdays":pdays,
 "previous":previous,"poutcome":poutcome,"emp.var.rate":emp,"cons.price.idx":cpi,"cons.conf.idx":cci,
 "euribor3m":euribor,"nr.employed":employed}])
if st.button("Predict subscription likelihood",type="primary"):
 probability=float(model.predict_proba(row)[0,1])
 st.metric("Estimated subscription probability",f"{probability:.1%}")
 st.progress(min(max(probability,0),1))
 if probability>=threshold: st.success(f"Above validation-selected threshold ({threshold:.3f}) — prioritize for review.")
 else: st.info(f"Below validation-selected threshold ({threshold:.3f}).")
with st.expander("Model information"):
 st.write("Selected model:",meta["best_model"])
 st.write("Threshold:",threshold)
 st.write("Duration excluded because it is known only after the call.")
