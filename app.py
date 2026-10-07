import os
import sys
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ============================================================
# PATH SETUP
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from src.predict import prepare_input


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Credit Risk Checker",
    page_icon="💳",
    layout="wide"
)


# ============================================================
# CSS
# ============================================================

st.markdown("""
<style>

.main-title {
    font-size: 36px;
    font-weight: 700;
    margin-bottom: 5px;
}

.subtitle {
    color: #777;
    font-size: 17px;
    margin-bottom: 25px;
}

.section-title {
    font-size: 22px;
    font-weight: 600;
    margin-top: 20px;
    margin-bottom: 10px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model_path = os.path.join(
        BASE_DIR,
        "models",
        "credit_scoring_model.joblib"
    )

    artifact = joblib.load(model_path)

    return artifact


artifact = load_model()

model = artifact["model"]
FEATURE_NAMES = artifact["feature_names"]
THRESHOLD = artifact["threshold"]


# ============================================================
# CREDIT SCORE
# ============================================================

def probability_to_score(probability):

    score = 850 - (float(probability) * 550)

    return int(max(300, min(850, score)))


# ============================================================
# RISK LEVEL
# ============================================================

def get_risk_level(score):

    if score >= 750:
        return "Low Risk", "🟢"

    elif score >= 670:
        return "Moderate Risk", "🔵"

    elif score >= 580:
        return "High Risk", "🟠"

    else:
        return "Very High Risk", "🔴"


# ============================================================
# RECOMMENDATION
# ============================================================

def get_recommendation(score):

    if score >= 750:

        return (
            "The applicant has a good payment profile. "
            "Continue making payments on time and keep "
            "credit usage under control."
        )

    elif score >= 670:

        return (
            "The applicant has a moderate payment risk. "
            "Maintaining regular payments and controlling "
            "monthly spending can improve the profile."
        )

    elif score >= 580:

        return (
            "The applicant shows an increased payment risk. "
            "Reducing outstanding balances and avoiding "
            "payment delays is recommended."
        )

    else:

        return (
            "The applicant shows a very high payment risk. "
            "Improving payment history and reducing "
            "outstanding balances is strongly recommended."
        )


# ============================================================
# SINGLE APPLICANT PREDICTION
# ============================================================

def predict_single(
    credit_limit,
    age,
    gender,
    education,
    marriage,
    payments,
    bills,
    pay_amounts
):

    data = {

        "LIMIT_BAL": [credit_limit],

        "SEX": [gender],

        "EDUCATION": [education],

        "MARRIAGE": [marriage],

        "AGE": [age],

        "PAY_1": [payments[0]],
        "PAY_2": [payments[1]],
        "PAY_3": [payments[2]],
        "PAY_4": [payments[3]],
        "PAY_5": [payments[4]],
        "PAY_6": [payments[5]],

        "BILL_AMT1": [bills[0]],
        "BILL_AMT2": [bills[1]],
        "BILL_AMT3": [bills[2]],
        "BILL_AMT4": [bills[3]],
        "BILL_AMT5": [bills[4]],
        "BILL_AMT6": [bills[5]],

        "PAY_AMT1": [pay_amounts[0]],
        "PAY_AMT2": [pay_amounts[1]],
        "PAY_AMT3": [pay_amounts[2]],
        "PAY_AMT4": [pay_amounts[3]],
        "PAY_AMT5": [pay_amounts[4]],
        "PAY_AMT6": [pay_amounts[5]]
    }

    df = pd.DataFrame(data)

    X = prepare_input(df)

    probability = model.predict_proba(X)[0, 1]

    prediction = int(probability >= THRESHOLD)

    return probability, prediction


# ============================================================
# CREDIT SCORE GAUGE
# ============================================================

def create_score_gauge(score):

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,

            title={
                "text": "Credit Score"
            },

            gauge={
                "axis": {
                    "range": [300, 850]
                },

                "bar": {
                    "thickness": 0.3
                },

                "steps": [
                    {
                        "range": [300, 580]
                    },
                    {
                        "range": [580, 670]
                    },
                    {
                        "range": [670, 750]
                    },
                    {
                        "range": [750, 850]
                    }
                ],

                "threshold": {
                    "line": {
                        "width": 4
                    },

                    "value": score
                }
            }
        )
    )

    fig.update_layout(
        height=350,
        margin=dict(
            l=20,
            r=20,
            t=60,
            b=20
        )
    )

    return fig


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">💳 Credit Risk Checker</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Check the likelihood of missing the next credit card payment.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Choose an option",
    [
        "👤 Check Applicant",
        "📁 Check Multiple Applicants",
        "⚡ What-if Analysis"
    ]
)


# ============================================================
# CHECK APPLICANT
# ============================================================

if page == "👤 Check Applicant":

    # ========================================================
    # APPLICANT INFORMATION
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        'Applicant Information'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        credit_limit = st.number_input(
            "Credit Limit",
            min_value=1000,
            max_value=1000000,
            value=200000,
            step=1000
        )

    with col2:

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=40
        )

    with col3:

        gender_label = st.selectbox(
            "Gender",
            [
                "Male",
                "Female"
            ]
        )

        gender = {
            "Male": 1,
            "Female": 2
        }[gender_label]


    # ========================================================
    # EDUCATION + MARITAL STATUS
    # ========================================================

    col1, col2 = st.columns(2)

    with col1:

        education_label = st.selectbox(
            "Education",
            [
                "Graduate School",
                "University",
                "High School",
                "Other"
            ]
        )

        education = {
            "Graduate School": 1,
            "University": 2,
            "High School": 3,
            "Other": 4
        }[education_label]


    with col2:

        marriage_label = st.selectbox(
            "Marital Status",
            [
                "Married",
                "Single",
                "Other"
            ]
        )

        marriage = {
            "Married": 1,
            "Single": 2,
            "Other": 3
        }[marriage_label]


    # ========================================================
    # PAYMENT HISTORY
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        'Payment History'
        '</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "Select the payment status for each of the last 6 months."
    )

    # UCI dataset mapping:
    #
    # -2 = No consumption
    # -1 = Paid duly / on time
    #  0 = Revolving credit
    #  1 = 1 month delay
    #  2 = 2 months delay
    #  3 = 3 months delay
    #  4 = 4 months delay
    #  5 = 5 months delay
    #  6 = 6+ months delay

    payment_options = {

        "Paid on time": -1,

        "No consumption": -2,

        "Revolving credit": 0,

        "1 month delay": 1,

        "2 months delay": 2,

        "3 months delay": 3,

        "4 months delay": 4,

        "5 months delay": 5,

        "6+ months delay": 6
    }

    month_labels = [
        "Latest Month",
        "1 Month Ago",
        "2 Months Ago",
        "3 Months Ago",
        "4 Months Ago",
        "5 Months Ago"
    ]

    payments = []

    cols = st.columns(3)

    for i in range(6):

        with cols[i % 3]:

            selected_payment = st.selectbox(
                month_labels[i],
                list(payment_options.keys()),
                key=f"payment_{i}"
            )

            payments.append(
                payment_options[selected_payment]
            )


    # ========================================================
    # RECENT BILLS
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        'Recent Bills'
        '</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "Enter the bill amount for each of the last 6 months."
    )

    bills = []

    cols = st.columns(3)

    for i in range(6):

        with cols[i % 3]:

            bill = st.number_input(
                month_labels[i],
                min_value=0.0,
                value=5000.0,
                step=500.0,
                key=f"bill_{i}"
            )

            bills.append(bill)


    # ========================================================
    # BILL WARNING
    # ========================================================

    if any(
        bill > credit_limit
        for bill in bills
    ):

        st.warning(
            "⚠️ One or more bills are higher than "
            "the credit limit. Please check the "
            "entered amount."
        )


    # ========================================================
    # RECENT PAYMENTS
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        'Recent Payments'
        '</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "Enter how much was paid during each month."
    )

    pay_amounts = []

    cols = st.columns(3)

    for i in range(6):

        with cols[i % 3]:

            payment_amount = st.number_input(
                month_labels[i],
                min_value=0.0,
                value=5000.0,
                step=500.0,
                key=f"pay_{i}"
            )

            pay_amounts.append(
                payment_amount
            )


    # ========================================================
    # CHECK BUTTON
    # ========================================================

    st.markdown("---")

    check_button = st.button(
        "🔍 Check Credit Risk",
        type="primary",
        use_container_width=True
    )


    # ========================================================
    # SINGLE APPLICANT RESULT
    # ========================================================

    if check_button:

        try:

            probability, prediction = predict_single(
                credit_limit,
                age,
                gender,
                education,
                marriage,
                payments,
                bills,
                pay_amounts
            )

            score = probability_to_score(
                probability
            )

            risk, risk_icon = get_risk_level(
                score
            )

            recommendation = get_recommendation(
                score
            )


            st.markdown("---")

            st.markdown(
                '<div class="section-title">'
                'Result'
                '</div>',
                unsafe_allow_html=True
            )

            col1, col2 = st.columns(2)

            with col1:

                st.plotly_chart(
                    create_score_gauge(score),
                    use_container_width=True
                )


            with col2:

                st.metric(
                    "Chance of Missing Next Payment",
                    f"{probability * 100:.1f}%"
                )

                st.metric(
                    "Credit Score",
                    score
                )

                st.markdown(
                    f"### {risk_icon} {risk}"
                )

                st.info(
                    recommendation
                )


            # =================================================
            # WHY THIS RESULT?
            # =================================================

            st.markdown("---")

            st.markdown(
                '<div class="section-title">'
                'Why this result?'
                '</div>',
                unsafe_allow_html=True
            )

            average_bill = np.mean(bills)

            credit_usage = (
                average_bill / credit_limit
            ) * 100

            longest_delay = max(
                max(payments),
                0
            )

            months_with_delay = sum(
                1
                for payment in payments
                if payment > 0
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Credit Usage",
                    f"{credit_usage:.1f}%"
                )

            with col2:

                st.metric(
                    "Longest Payment Delay",
                    f"{longest_delay} month(s)"
                )

            with col3:

                st.metric(
                    "Months With Payment Delay",
                    months_with_delay
                )


        except Exception as e:

            st.error(
                f"Unable to calculate the result: {e}"
            )


# ============================================================
# CHECK MULTIPLE APPLICANTS
# ============================================================

elif page == "📁 Check Multiple Applicants":

    st.markdown(
        '<div class="section-title">'
        'Check Multiple Applicants'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Upload a CSV file containing multiple applicants."
    )

    st.caption(
        "The app will calculate the risk for every applicant "
        "in the file."
    )


    # ========================================================
    # UPLOAD CSV
    # ========================================================

    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"]
    )


    if uploaded_file is not None:

        try:

            df = pd.read_csv(
                uploaded_file
            )


            # =================================================
            # SHOW INPUT DATA
            # =================================================

            st.subheader(
                "Uploaded Applicants"
            )

            st.dataframe(
                df,
                use_container_width=True,
                height=300
            )


            # =================================================
            # CHECK ALL BUTTON
            # =================================================

            if st.button(
                "🔍 Check All Applicants",
                type="primary",
                use_container_width=True
            ):

                # ---------------------------------------------
                # PREPARE ALL APPLICANTS
                # ---------------------------------------------

                X = prepare_input(df)


                # ---------------------------------------------
                # PREDICT ALL ROWS AT ONCE
                # ---------------------------------------------

                probabilities = model.predict_proba(
                    X
                )[:, 1]


                # ---------------------------------------------
                # CREATE RESULT TABLE
                # ---------------------------------------------

                result_data = []

                for i, probability in enumerate(
                    probabilities
                ):

                    score = probability_to_score(
                        probability
                    )

                    risk, risk_icon = get_risk_level(
                        score
                    )

                    recommendation = get_recommendation(
                        score
                    )

                    result_data.append({

                        "Applicant": i + 1,

                        "Chance of Missing Next Payment":
                            f"{probability * 100:.1f}%",

                        "Credit Score":
                            score,

                        "Payment Risk":
                            f"{risk_icon} {risk}",

                        "Recommendation":
                            recommendation
                    })


                results = pd.DataFrame(
                    result_data
                )


                # =================================================
                # SUCCESS
                # =================================================

                st.success(
                    f"Successfully checked "
                    f"{len(results)} applicants."
                )


                # =================================================
                # RESULTS
                # =================================================

                st.markdown("---")

                st.markdown(
                    '<div class="section-title">'
                    'Results'
                    '</div>',
                    unsafe_allow_html=True
                )


                st.dataframe(
                    results,
                    use_container_width=True,
                    hide_index=True,
                    height=400
                )


                # =================================================
                # SUMMARY
                # =================================================

                st.markdown("---")

                st.subheader(
                    "Summary"
                )

                low_count = sum(
                    results["Payment Risk"].str.contains(
                        "Low Risk"
                    )
                )

                moderate_count = sum(
                    results["Payment Risk"].str.contains(
                        "Moderate Risk"
                    )
                )

                high_count = sum(
                    results["Payment Risk"].str.contains(
                        "High Risk"
                    )
                    & ~results["Payment Risk"].str.contains(
                        "Very High Risk"
                    )
                )

                very_high_count = sum(
                    results["Payment Risk"].str.contains(
                        "Very High Risk"
                    )
                )


                col1, col2, col3, col4 = st.columns(4)

                with col1:

                    st.metric(
                        "Low Risk",
                        low_count
                    )

                with col2:

                    st.metric(
                        "Moderate Risk",
                        moderate_count
                    )

                with col3:

                    st.metric(
                        "High Risk",
                        high_count
                    )

                with col4:

                    st.metric(
                        "Very High Risk",
                        very_high_count
                    )


        except Exception as e:

            st.error(
                f"Unable to process the file: {e}"
            )


# ============================================================
# WHAT-IF ANALYSIS
# ============================================================


elif page == "⚡ What-if Analysis":

    st.markdown(
        '<div class="section-title">What-if Analysis</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Change the values and check how the situation changes."
    )

    credit_limit = st.number_input(
        "Credit Limit",
        min_value=1000,
        max_value=1000000,
        value=200000,
        step=1000
    )

    average_bill = st.number_input(
        "Average Monthly Bill",
        min_value=0.0,
        value=10000.0,
        step=500.0
    )

    longest_delay = st.selectbox(
        "Longest Payment Delay",
        [
            "No delay",
            "1 month",
            "2 months",
            "3 months",
            "4 months",
            "5 months",
            "6+ months"
        ]
    )

    months_with_delay = st.slider(
        "Months With Payment Delay",
        min_value=0,
        max_value=6,
        value=0
    )

    # --------------------------------------------------------
    # WARNING
    # --------------------------------------------------------

    if average_bill > credit_limit:

        st.warning(
            "⚠️ The average monthly bill is higher "
            "than the credit limit."
        )

    # --------------------------------------------------------
    # SEE RESULT BUTTON
    # --------------------------------------------------------

    if st.button(
        "🔍 See Result",
        type="primary",
        use_container_width=True
    ):

        usage = (
            average_bill / credit_limit
        ) * 100

        st.markdown("---")

        st.markdown(
            '<div class="section-title">Result</div>',
            unsafe_allow_html=True
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Credit Usage",
                f"{usage:.1f}%"
            )

        with col2:
            st.metric(
                "Longest Payment Delay",
                longest_delay
            )

        with col3:
            st.metric(
                "Months With Payment Delay",
                months_with_delay
            )

        # Simple interpretation

        if (
            months_with_delay == 0
            and longest_delay == "No delay"
            and usage <= 30
        ):

            st.success(
                "🟢 This is a relatively healthy "
                "payment situation."
            )

        elif (
            months_with_delay <= 2
            and usage <= 70
        ):

            st.info(
                "🔵 This situation shows some "
                "payment or credit usage risk."
            )

        else:

            st.warning(
                "🔴 This situation shows higher "
                "payment risk."
            )