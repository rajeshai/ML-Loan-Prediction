from pycaret.classification import load_model, predict_model
import streamlit as st
import pandas as pd
model=load_model('Logistic regression')
    
def run():
    
    def predict(model, input_df):
       predictions_df=predict_model(estimator=model,data=input_df)
       predictions=predictions_df['Label'][0]
       return predictions
    st.title("Welcome to Dream Finance Housing Company")
    st.header("Please choose fill in the choices to get your Loan Eligibility")
    Gender=st.selectbox('Gender',['Select','Male', 'Female'])
    Married=st.selectbox('Married',['Select','Yes', 'No'])
    Dependents=st.selectbox('Dependents',['Select','0','1','2','3+'])
    Education=st.selectbox('Education',['Select','Graduate', 'Not Graduate'])
    Self_Employed=st.selectbox('Self Employed',['Select','Yes', 'No'])
    ApplicantIncome=st.number_input('Applicant Income', min_value=1.0, max_value=100000.0)
    CoapplicantIncome=st.number_input('Coapplicant Income', min_value=0.0, max_value=100000.0)
    LoanAmount=st.number_input('Loan Amount(In Hundreds)', min_value=1)
    Loan_Amount_Term=st.selectbox('Term of Loan Amount(In Months)',['Select','60','120','180','240','300','360'])
    Credit_History=st.selectbox('Credit History',['Select','0','1'])
    Property_Area=st.selectbox('Select your Area Category',['Select','Urban','Semiurban','Rural'])
    output=""
    input_dict={'Gender':Gender, 
                'Married': Married, 
                'Dependents':Dependents, 
                'Education':Education,
                'Self_Employed':Self_Employed,
               'ApplicantIncome':ApplicantIncome,
               'CoapplicantIncome':CoapplicantIncome,
               'LoanAmount':LoanAmount,
               'Loan_Amount_Term':Loan_Amount_Term,
               'Credit_History':Credit_History,
               'Property_Area':Property_Area}
    input_df=pd.DataFrame([input_dict])
    if st.button("Check"):
        output=predict(model=model, input_df=input_df)
        output=str(output)
        if output=='Y':
            st.success('Congrats! You are eligible to avail the Loan')
            #st.success('Your Loan Eligibility Status {}'.format(output))
        else:
            st.success("Sorry!! You are not eligible to avail the loan")


if __name__=='__main__':
    run()
