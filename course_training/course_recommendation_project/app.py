import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# =========================================================
# PAGE SETTINGS
# =========================================================

st.set_page_config(
    page_title="Course Recommendation System",
    page_icon="🎓",
    layout="wide"
)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    model_path = Path(__file__).parent / "course_recommendation_model.pkl"

    if not model_path.exists():

        st.error(
            "❌ Model file not found.\n\n"
            "Make sure course_recommendation_model.pkl "
            "is in the same folder as app.py."
        )

        st.stop()

    try:

        data = joblib.load(model_path)

        return (
            data["model"],
            data["preprocessor"],
            data["label_encoders"],
            data["target_columns"],
            data["fields"]
        )

    except Exception as e:

        st.error("❌ Could not load the trained model.")

        st.code(str(e))

        st.info(
            "Make sure the model was trained and saved "
            "with compatible Python and scikit-learn versions."
        )

        st.stop()


model, preprocessor, label_encoders, target_columns, fields = load_model()


# =========================================================
# TITLE
# =========================================================

st.title("🎓 Course Recommendation System")

st.write(
    "Enter your field, hobbies, grades and aptitude scores "
    "to get personalized Top 5 course recommendations."
)


# =========================================================
# FIELD
# =========================================================

st.subheader("🎓 Select Your Field")

field = st.selectbox(
    "Choose your field",
    fields,
    key="CRS_FIELD_SELECTION_001"
)


# =========================================================
# STUDENT DATA
# =========================================================

student = {
    "field_filter": field
}


# =========================================================
# HOBBIES
# =========================================================

st.subheader("🎯 Hobbies")

hobby_columns = [

    "hobby_coding",
    "hobby_gaming",
    "hobby_reading",
    "hobby_writing",
    "hobby_music",
    "hobby_drawing_art",
    "hobby_sports",
    "hobby_cooking",
    "hobby_photography",
    "hobby_travel",
    "hobby_science_experiments",
    "hobby_volunteering",
    "hobby_debating",
    "hobby_robotics",
    "hobby_fashion",
    "hobby_business_trading"

]


cols = st.columns(4)


for i, hobby in enumerate(hobby_columns):

    hobby_name = (
        hobby
        .replace("hobby_", "")
        .replace("_", " ")
        .title()
    )

    with cols[i % 4]:

        selected = st.checkbox(
            hobby_name,
            value=False,
            key=f"CRS_HOBBY_{hobby}"
        )

    student[hobby] = 1 if selected else 0


# =========================================================
# GRADES
# =========================================================

st.subheader("📚 Grades")

grade_columns = [

    "grade_math",
    "grade_science",
    "grade_lang",
    "grade_social",
    "grade_cs"

]


grade_names = {

    "grade_math": "Math",
    "grade_science": "Science",
    "grade_lang": "Language",
    "grade_social": "Social",
    "grade_cs": "Computer Science"

}


cols = st.columns(5)


for i, grade_col in enumerate(grade_columns):

    with cols[i]:

        student[grade_col] = st.number_input(

            grade_names[grade_col],

            min_value=0.0,

            max_value=100.0,

            value=50.0,

            step=1.0,

            key=f"CRS_GRADE_{grade_col}"

        )


# =========================================================
# APTITUDE SCORES
# =========================================================

st.subheader("🧠 Aptitude Scores")

score_columns = [

    "score_analytical",
    "score_numeric",
    "score_verbal",
    "score_creative",
    "score_social"

]


score_names = {

    "score_analytical": "Analytical",
    "score_numeric": "Numeric",
    "score_verbal": "Verbal",
    "score_creative": "Creative",
    "score_social": "Social"

}


cols = st.columns(5)


for i, score_col in enumerate(score_columns):

    with cols[i]:

        student[score_col] = st.number_input(

            score_names[score_col],

            min_value=0.0,

            max_value=100.0,

            value=50.0,

            step=1.0,

            key=f"CRS_APTITUDE_{score_col}"

        )


# =========================================================
# PREDICTION BUTTON
# =========================================================

st.divider()


predict = st.button(

    "🔮 Predict Top 5 Courses",

    use_container_width=True,

    type="primary",

    key="CRS_PREDICT_BUTTON_001"

)


# =========================================================
# PREDICTION
# =========================================================

if predict:

    try:

        # -------------------------------------------------
        # Create DataFrame
        # -------------------------------------------------

        student_df = pd.DataFrame([student])


        # -------------------------------------------------
        # Expected input columns
        # -------------------------------------------------

        expected_columns = (

            ["field_filter"]

            + hobby_columns

            + grade_columns

            + score_columns

        )


        # -------------------------------------------------
        # Check missing columns
        # -------------------------------------------------

        missing_columns = [

            col

            for col in expected_columns

            if col not in student_df.columns

        ]


        if missing_columns:

            st.error("❌ Missing input columns:")

            st.write(missing_columns)

            st.stop()


        # -------------------------------------------------
        # Correct column order
        # -------------------------------------------------

        student_df = student_df[expected_columns]


        # -------------------------------------------------
        # Preprocess
        # -------------------------------------------------

        processed = preprocessor.transform(student_df)


        # -------------------------------------------------
        # Prediction
        # -------------------------------------------------

        prediction = model.predict(processed)


        # -------------------------------------------------
        # Convert prediction to 2D
        # -------------------------------------------------

        if hasattr(prediction, "ndim"):

            if prediction.ndim == 1:

                prediction = prediction.reshape(1, -1)


        # -------------------------------------------------
        # Check prediction size
        # -------------------------------------------------

        if prediction.shape[1] != len(target_columns):

            st.error(

                "❌ Prediction output does not match "
                "the number of course recommendation columns."

            )

            st.write(
                "Prediction shape:",
                prediction.shape
            )

            st.write(
                "Number of target columns:",
                len(target_columns)
            )

            st.stop()


        # -------------------------------------------------
        # Display recommendations
        # -------------------------------------------------

        st.subheader("🏆 Recommended Top 5 Courses")


        for i, target_col in enumerate(target_columns):

            predicted_value = prediction[0, i]


            # -------------------------------------------------
            # Decode course
            # -------------------------------------------------

            encoder = label_encoders[target_col]


            try:

                course = encoder.inverse_transform(
                    [int(predicted_value)]
                )[0]

            except Exception:

                course = str(predicted_value)


            # -------------------------------------------------
            # Ranking display
            # -------------------------------------------------

            if i == 0:

                st.success(
                    f"🥇 **1. {course}**"
                )

            elif i == 1:

                st.info(
                    f"🥈 **2. {course}**"
                )

            elif i == 2:

                st.warning(
                    f"🥉 **3. {course}**"
                )

            else:

                st.write(
                    f"### {i + 1}. {course}"
                )


        # -------------------------------------------------
        # Input Summary
        # -------------------------------------------------

        with st.expander("📋 View Your Input"):

            st.dataframe(
                student_df,
                use_container_width=True
            )


    except Exception as e:

        st.error("❌ Prediction failed.")

        st.code(str(e))

        st.info(

            "The model, preprocessor, input columns or "
            "label encoders may not match the training code."

        )