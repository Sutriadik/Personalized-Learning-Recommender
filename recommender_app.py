import streamlit as st
import pandas as pd
import time
import backend as backend

from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid import GridUpdateMode, DataReturnMode

# Basic webpage setup
st.set_page_config(
   page_title="Course Recommender System",
   layout="wide",
   initial_sidebar_state="expanded",
)


# ------- Functions ------
# Load datasets
@st.cache_data
def load_ratings():
    return backend.load_ratings()


@st.cache_data
def load_course_sims():
    return backend.load_course_sims()


@st.cache_data
def load_courses():
    return backend.load_courses()


@st.cache_data
def load_bow():
    return backend.load_bow()


# Initialize the app by first loading datasets
def init__recommender_app():

    with st.spinner('Loading datasets...'):
        ratings_df = load_ratings()
        sim_df = load_course_sims()
        course_df = load_courses()
        course_bow_df = load_bow()

    # Select courses
    st.success('Datasets loaded successfully...')

    st.markdown("""---""")
    st.subheader("Select courses that you have audited or completed: ")

    # Build an interactive table for `course_df`
    gb = GridOptionsBuilder.from_dataframe(course_df)
    gb.configure_default_column(enablePivot=True, enableValue=True, enableRowGroup=True)
    gb.configure_selection(selection_mode="multiple", use_checkbox=True)
    gb.configure_side_bar()
    grid_options = gb.build()

    # Create a grid response
    response = AgGrid(
        course_df,
        gridOptions=grid_options,
        enable_enterprise_modules=True,
        update_mode=GridUpdateMode.MODEL_CHANGED,
        data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
        fit_columns_on_grid_load=False,
    )

    results = pd.DataFrame(response["selected_rows"], columns=['COURSE_ID', 'TITLE', 'DESCRIPTION'])
    results = results[['COURSE_ID', 'TITLE']]
    st.subheader("Your courses: ")
    st.table(results)
    return results


def train(model_name, params):
    # Course Similarity and User Profile need no training
    no_train_models = [backend.models[0], backend.models[1]]
    if model_name in no_train_models:
        st.sidebar.info(f'"{model_name}" does not require a training step.')
        return

    with st.spinner('Training...'):
        time.sleep(0.5)
        backend.train(model_name, params)
    st.success('Done!')


def predict(model_name, user_ids, params):
    res = None
    with st.spinner('Generating course recommendations: '):
        time.sleep(0.5)
        res = backend.predict(model_name, user_ids, params)
    st.success('Recommendations generated!')
    return res


# ------ UI ------
# Sidebar
st.sidebar.title('Personalized Learning Recommender')
# Initialize the app
selected_courses_df = init__recommender_app()

# Model selection selectbox
st.sidebar.subheader('1. Select recommendation models')
model_selection = st.sidebar.selectbox(
    "Select model:",
    backend.models
)

# Hyper-parameters for each model
params = {}
st.sidebar.subheader('2. Tune Hyper-parameters: ')

# Course Similarity
if model_selection == backend.models[0]:
    top_courses = st.sidebar.slider('Top courses',
                                    min_value=0, max_value=100,
                                    value=10, step=1)
    course_sim_threshold = st.sidebar.slider('Course Similarity Threshold %',
                                             min_value=0, max_value=100,
                                             value=50, step=10)
    params['top_courses'] = top_courses
    params['sim_threshold'] = course_sim_threshold

# User Profile
elif model_selection == backend.models[1]:
    profile_sim_threshold = st.sidebar.slider('User Profile Similarity Threshold %',
                                              min_value=0, max_value=100,
                                              value=10, step=5)
    params['profile_sim_threshold'] = profile_sim_threshold

# Clustering
elif model_selection == backend.models[2]:
    cluster_no = st.sidebar.slider('Number of Clusters',
                                   min_value=2, max_value=50,
                                   value=20, step=1)
    params['cluster_no'] = cluster_no

# Clustering with PCA
elif model_selection == backend.models[3]:
    cluster_no = st.sidebar.slider('Number of Clusters',
                                   min_value=2, max_value=50,
                                   value=20, step=1)
    params['cluster_no'] = cluster_no

# KNN
elif model_selection == backend.models[4]:
    n_neighbors = st.sidebar.slider('Number of Neighbors (K)',
                                    min_value=1, max_value=50,
                                    value=10, step=1)
    params['n_neighbors'] = n_neighbors

# NMF
elif model_selection == backend.models[5]:
    n_components = st.sidebar.slider('Number of Latent Factors',
                                     min_value=2, max_value=50,
                                     value=15, step=1)
    params['n_components'] = n_components

# Neural Network
elif model_selection == backend.models[6]:
    n_components = st.sidebar.slider('Number of Embedding Dimensions',
                                     min_value=2, max_value=50,
                                     value=15, step=1)
    params['n_components'] = n_components

# Regression with Embedding Features
elif model_selection == backend.models[7]:
    n_components = st.sidebar.slider('Number of Embedding Dimensions',
                                     min_value=2, max_value=50,
                                     value=15, step=1)
    params['n_components'] = n_components

# Classification with Embedding Features
elif model_selection == backend.models[8]:
    n_components = st.sidebar.slider('Number of Embedding Dimensions',
                                     min_value=2, max_value=50,
                                     value=15, step=1)
    params['n_components'] = n_components


# Training
st.sidebar.subheader('3. Training: ')
training_button = st.sidebar.button("Train Model")
training_text = st.sidebar.text('')
# Start training process
if training_button:
    train(model_selection, params)


# Prediction
st.sidebar.subheader('4. Prediction')
# Start prediction process
pred_button = st.sidebar.button("Recommend New Courses")
if pred_button and selected_courses_df.shape[0] > 0:
    # Create a new id for current user session
    new_id = backend.add_new_ratings(selected_courses_df['COURSE_ID'].values)
    user_ids = [new_id]
    res_df = predict(model_selection, user_ids, params)
    res_df = res_df[['COURSE_ID', 'SCORE']]
    course_df = load_courses()
    res_df = pd.merge(res_df, course_df, on=["COURSE_ID"]).drop('COURSE_ID', axis=1)

    # Format display
    res_df['SCORE'] = res_df['SCORE'].round(4)
    res_df = res_df.reset_index(drop=True)
    res_df.index += 1
    res_df.index.name = "No."
    res_df = res_df.rename(columns={'TITLE': 'Course Title', 'SCORE': 'Score', 'DESCRIPTION': 'Description'})

    st.markdown("### Recommended Courses")
    st.dataframe(
        res_df,
        use_container_width=True,
        column_config={
            "No.":          st.column_config.NumberColumn(width="small"),
            "Score":        st.column_config.NumberColumn(format="%.4f", width="small"),
            "Course Title": st.column_config.TextColumn(width="medium"),
            "Description":  st.column_config.TextColumn(width="large"),
        }
    )
