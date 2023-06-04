import base64

import joblib
import pandas
import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

import warnings

warnings.filterwarnings('ignore')

st.title('Consumer buying behavior analytics')


def create_download_link(df, title="Download CSV output", filename="predicted.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = f'<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    return html


def process_data(data):

    initial_data = data.copy()
    data = data.drop(['z_cost_to_contact', 'z_profit'], axis=1)
    data.head()

    data['salary'] = data['salary'].fillna(data['salary'].mean())
    data.isna().any()

    data['relationship_status'] = data['relationship_status'].replace(['Married', 'Together'], 'relationship')
    data['relationship_status'] = data['relationship_status'].replace(['Divorced', 'Widow', 'Alone', 'YOLO', 'Absurd'],
                                                                      'Single')

    product_data = []
    for i in range(0, len(data)):
        productdata = [data['wines_amount'][i], data['fruits_amount'][i],
                       data['meat_amount'][i], data['fish_amount'][i],
                       data['dessert_amount'][i], data['gold_amount'][i]]
        product_data.append(productdata)
    products_data = pandas.DataFrame(product_data, columns=['Wines', 'Fruits', 'Meat', 'Fish', 'Sweets', 'Gold'])
    products_data.head()

    data['Kids'] = data['kids_count'] + data['teens_count']
    data['Expenses'] = data['wines_amount'] + data['fruits_amount'] + data['meat_amount'] + data['fish_amount'] + data[
        'dessert_amount'] + data['gold_amount']
    data['total_accepted_campaings'] = data['is_campaign_accepted_1'] + data['is_campaign_accepted_2'] + data[
        'is_campaign_accepted_3'] + data['is_campaign_accepted_4'] + data['is_campaign_accepted_5'] + data[
                                           'last_campaign_accepted']
    data['total_purchases'] = data['web_purchases'] + data['catalog_purchases'] + data['instore_purchases'] + data[
        'deal_purchases']

    col_del = ["is_campaign_accepted_1", "is_campaign_accepted_2", "is_campaign_accepted_3", "is_campaign_accepted_4",
               "is_campaign_accepted_5", "last_campaign_accepted", "web_visits_per_month", "web_purchases",
               "catalog_purchases", "instore_purchases", "deal_purchases", "kids_count", "teens_count", "wines_amount",
               "fruits_amount", "meat_amount", "fish_amount", "dessert_amount", "gold_amount"]
    data = data.drop(columns=col_del, axis=1)
    data.head()

    data['Age'] = 2015 - data["birth_year"]

    data['customer_since'] = pandas.to_datetime(data['customer_since'], format='%d-%m-%Y')
    data['first_day'] = '01-01-2015'
    data['first_day'] = pandas.to_datetime(data['first_day'], format='%d-%m-%Y')
    data['day_engaged'] = (data['first_day'] - data['customer_since']).dt.days

    data = data.drop(
        columns=["id", "customer_since", "first_day", "birth_year", "customer_since", "last_bought", "complained"],
        axis=1)

    # Analyzing data

    st.subheader('Data Statistics')
    st.write(data.describe())

    cate = []
    for i in data.columns:
        if (data[i].dtypes == "object"):
            cate.append(i)
    st.subheader('Categorical Features')
    st.write(cate)

    lbl_encode = LabelEncoder()
    for i in cate:
        data[i] = data[[i]].apply(lbl_encode.fit_transform)

    st.subheader('Data Statistics')
    st.write(data.describe())

    data_copy = data.copy()

    st.subheader('Data head')
    st.write(data_copy.head(3))

    st.subheader('Buying behavior prediction')

    data_copied = data_copy.copy()
    x = data_copied.copy()
    sc = StandardScaler()
    log_reg_loaded = joblib.load('log_reg_buying_behavior_prediction.pkl')
    st.write(x)
    x = sc.fit_transform(x)
    st.write(x)
    y_predicted = log_reg_loaded.predict(x)
    df_predicted = pd.DataFrame(y_predicted, columns=['predicted'])
    st.write('Output: ')
    st.write('1 and 2 in "predicted" column stand for will not buy and will buy respectively')
    st.write(df_predicted)
    initial_data = initial_data.join(df_predicted['predicted'])
    download_link = create_download_link(initial_data, "Download predictions", "predicted.csv")

    # Display the download link
    st.markdown(download_link, unsafe_allow_html=True)


# Load data
def load_data():
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv", )
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file, sep="\t")
        return data


loaded_data = load_data()

if loaded_data is not None:
    process_data(loaded_data)
else:
    st.write("Please upload a file.")
