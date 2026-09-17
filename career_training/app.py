
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path


# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="Career Recommendation System",
    page_icon="🎓",
    layout="wide"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model_path = Path(__file__).parent / "career_ranking_model.pkl"

    if not model_path.exists():

        st.error(
            "❌ career_ranking_model.pkl not found."
        )

        st.stop()

    return joblib.load(model_path)


data = load_model()

ranking_model = data["model"]
feature_columns = data["feature_columns"]


# ============================================================
# TITLE
# ============================================================

st.title("🎓 Career Recommendation System")

st.write(
    "Enter your field, hobbies, academic grades and aptitude "
    "scores to receive your Top 5 career recommendations."
)


# ============================================================
# FIELD
# ============================================================

st.subheader("🎓 Select Your Field")

fields = [
    "Engineering",
    "Medical & Healthcare",
    "Business & Finance",
    "Science & Research",
    "Arts & Design",
    "Computer Science & IT",
    "Education",
    "Law & Government",
    "Media & Communication",
    "Sports & Fitness"
]

field = st.selectbox(
    "Choose your field",
    fields
)


# ============================================================
# STUDENT DATA
# ============================================================

student = {}


# ============================================================
# HOBBIES
# ============================================================

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

    name = (
        hobby
        .replace("hobby_", "")
        .replace("_", " ")
        .title()
    )

    with cols[i % 4]:

        selected = st.checkbox(name)

    student[hobby] = 1 if selected else 0


# ============================================================
# ACADEMIC GRADES
# ============================================================

st.subheader("📚 Academic Grades")

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
    "grade_social": "Social Science",
    "grade_cs": "Computer Science"

}


cols = st.columns(5)


for i, grade in enumerate(grade_columns):

    with cols[i]:

        student[grade] = st.number_input(

            grade_names[grade],

            min_value=0.0,
            max_value=100.0,
            value=50.0,
            step=1.0

        )


# ============================================================
# APTITUDE SCORES
# ============================================================

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


for i, score in enumerate(score_columns):

    with cols[i]:

        student[score] = st.number_input(

            score_names[score],

            min_value=0.0,
            max_value=100.0,
            value=50.0,
            step=1.0

        )


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_features(student):

    df = pd.DataFrame([student])


    # --------------------------------------------------------
    # Academic
    # --------------------------------------------------------

    df["academic_average"] = (

        df["grade_math"]
        + df["grade_science"]
        + df["grade_lang"]
        + df["grade_social"]
        + df["grade_cs"]

    ) / 5


    df["stem_average"] = (

        df["grade_math"]
        + df["grade_science"]
        + df["grade_cs"]

    ) / 3


    df["language_social_average"] = (

        df["grade_lang"]
        + df["grade_social"]

    ) / 2


    # --------------------------------------------------------
    # Aptitude
    # --------------------------------------------------------

    df["aptitude_average"] = (

        df["score_analytical"]
        + df["score_numeric"]
        + df["score_verbal"]
        + df["score_creative"]
        + df["score_social"]

    ) / 5


    df["logical_strength"] = (

        df["score_analytical"]
        + df["score_numeric"]

    ) / 2


    df["communication_strength"] = (

        df["score_verbal"]
        + df["score_social"]

    ) / 2


    df["creative_strength"] = (

        df["score_creative"]
        + df["score_verbal"]

    ) / 2


    # --------------------------------------------------------
    # Career-related features
    # --------------------------------------------------------

    df["technology_strength"] = (

        df["grade_cs"]
        + df["score_analytical"]
        + df["score_numeric"]
        + df["hobby_coding"] * 100
        + df["hobby_robotics"] * 100

    ) / 5


    df["science_strength"] = (

        df["grade_science"]
        + df["score_analytical"]
        + df["score_numeric"]
        + df["hobby_science_experiments"] * 100

    ) / 4


    df["business_strength"] = (

        df["score_numeric"]
        + df["score_social"]
        + df["score_verbal"]
        + df["hobby_business_trading"] * 100

    ) / 4


    df["creative_strength_combined"] = (

        df["score_creative"]
        + df["grade_lang"]
        + df["hobby_drawing_art"] * 100
        + df["hobby_music"] * 100
        + df["hobby_writing"] * 100
        + df["hobby_fashion"] * 100

    ) / 6


    df["social_strength"] = (

        df["score_social"]
        + df["score_verbal"]
        + df["grade_social"]
        + df["hobby_debating"] * 100
        + df["hobby_volunteering"] * 100

    ) / 5


    # --------------------------------------------------------
    # Hobby count
    # --------------------------------------------------------

    df["hobby_count"] = df[hobby_columns].sum(axis=1)


    # --------------------------------------------------------
    # Overall strength
    # --------------------------------------------------------

    df["overall_strength"] = (

        df["academic_average"]
        + df["aptitude_average"]

    ) / 2


    return df


# ============================================================
# FIELD-BASED CAREER BOOST
# ============================================================
#
# The current ranking model was trained with 40 numerical
# features and does NOT contain field_filter.
#
# Therefore, we apply a small field compatibility boost here.
#
# This makes changing the selected field actually influence
# the final ranking.
# ============================================================

field_keywords = {

    "Engineering": [

        "engineer",
        "architect",
        "technician"

    ],

    "Medical & Healthcare": [

        "doctor",
        "nurse",
        "medical",
        "physiotherapist",
        "psychologist",
        "health",
        "laboratory"

    ],

    "Business & Finance": [

        "accountant",
        "manager",
        "marketing",
        "business",
        "finance",
        "financial",
        "trading",
        "operations",
        "human resource",
        "hr",
        "economist"

    ],

    "Science & Research": [

        "scientist",
        "research",
        "physicist",
        "chemist",
        "biologist",
        "astronomer",
        "mathematician",
        "statistician",
        "geologist",
        "oceanographer",
        "ocean",
        "space"

    ],

    "Arts & Design": [

        "designer",
        "artist",
        "animator",
        "fashion",
        "interior",
        "vfx",
        "illustrator",
        "drawing"

    ],

    "Computer Science & IT": [

        "software",
        "developer",
        "programmer",
        "data scientist",
        "data analyst",
        "machine learning",
        "ai engineer",
        "cloud engineer",
        "devops",
        "network engineer",
        "cyber",
        "forensics",
        "web developer",
        "game developer",
        "it project"

    ],

    "Education": [

        "teacher",
        "professor",
        "trainer",
        "education",
        "special education",
        "instructor"

    ],

    "Law & Government": [

        "lawyer",
        "advocate",
        "government",
        "officer",
        "naval",
        "coast guard",
        "crime",
        "forensic"

    ],

    "Media & Communication": [

        "journalist",
        "writer",
        "content",
        "editor",
        "film",
        "director",
        "media",
        "public relations",
        "social media",
        "video"

    ],

    "Sports & Fitness": [

        "coach",
        "fitness",
        "sports",
        "physiotherapist",
        "yoga",
        "athlete"

    ]

}


def apply_field_boost(results, selected_field):

    keywords = field_keywords.get(
        selected_field,
        []
    )

    def calculate_boost(career):

        career_text = str(career).lower()

        for keyword in keywords:

            if keyword.lower() in career_text:

                return 0.20

        return 0.0


    results["Field_Boost"] = results["Career"].apply(
        calculate_boost
    )


    results["Final_Score"] = (

        results["Score"]
        + results["Field_Boost"]

    )


    return results


# ============================================================
# PREDICTION
# ============================================================

st.divider()


if st.button(

    "🔮 Predict Top 5 Careers",

    use_container_width=True,

    type="primary"

):

    try:

        # ----------------------------------------------------
        # CREATE FEATURES
        # ----------------------------------------------------

        student_df = create_features(student)


        # ----------------------------------------------------
        # CHECK FEATURES
        # ----------------------------------------------------

        missing_features = [

            col

            for col in feature_columns

            if col not in student_df.columns

        ]


        if missing_features:

            st.error(
                "❌ Some model features are missing."
            )

            st.write(missing_features)

            st.stop()


        # ----------------------------------------------------
        # EXACT FEATURE ORDER
        # ----------------------------------------------------

        X = student_df[
            feature_columns
        ].copy()


        X_processed = X.values


        # ----------------------------------------------------
        # MODEL PROBABILITIES
        # ----------------------------------------------------

        probabilities = ranking_model.predict_proba(
            X_processed
        )[0]


        classes = ranking_model.classes_


        # ----------------------------------------------------
        # CREATE RESULTS
        # ----------------------------------------------------

        results = pd.DataFrame({

            "Career": classes,

            "Score": probabilities

        })


        # ====================================================
        # REMOVE NOT_RECOMMENDED
        # ====================================================

        results = results[

            results["Career"].astype(str).str.upper()
            != "NOT_RECOMMENDED"

        ].copy()


        # ----------------------------------------------------
        # NORMALIZE MODEL SCORES
        # ----------------------------------------------------

        total_score = results["Score"].sum()


        if total_score > 0:

            results["Score"] = (
                results["Score"]
                / total_score
            )


        # ----------------------------------------------------
        # APPLY FIELD BOOST
        # ----------------------------------------------------

        results = apply_field_boost(
            results,
            field
        )


        # ----------------------------------------------------
        # SORT FINAL RESULTS
        # ----------------------------------------------------

        results = results.sort_values(

            by="Final_Score",

            ascending=False

        ).reset_index(drop=True)


        # ----------------------------------------------------
        # NORMALIZE FINAL SCORES
        # ----------------------------------------------------

        total_final = results["Final_Score"].sum()


        if total_final > 0:

            results["Final_Score"] = (

                results["Final_Score"]
                / total_final

            )


        # ----------------------------------------------------
        # TOP 5
        # ----------------------------------------------------

        top_5 = results.head(5).copy()

        top_5 = top_5.reset_index(drop=True)


        # ====================================================
        # DISPLAY
        # ====================================================

        st.subheader(
            "🏆 Your Top 5 Career Recommendations"
        )


        st.write(

            f"Recommendations are based on your "
            f"**{field}** field, hobbies, grades and "
            f"aptitude scores."

        )


        # ====================================================
        # CAREER CARDS
        # ====================================================

        for i in range(len(top_5)):

            career = top_5.loc[
                i,
                "Career"
            ]

            score = top_5.loc[
                i,
                "Final_Score"
            ] * 100


            if i == 0:

                st.success(

                    f"🥇 **1. {career}**\n\n"
                    f"Suitability Score: **{score:.2f}%**"

                )


            elif i == 1:

                st.info(

                    f"🥈 **2. {career}**\n\n"
                    f"Suitability Score: **{score:.2f}%**"

                )


            elif i == 2:

                st.warning(

                    f"🥉 **3. {career}**\n\n"
                    f"Suitability Score: **{score:.2f}%**"

                )


            else:

                st.write(
                    f"### {i + 1}. {career}"
                )


                st.progress(

                    min(
                        float(
                            top_5.loc[
                                i,
                                "Final_Score"
                            ]
                        ),
                        1.0
                    )

                )


                st.write(

                    f"Suitability Score: "
                    f"**{score:.2f}%**"

                )


        # ====================================================
        # SUMMARY TABLE
        # ====================================================

        st.divider()

        st.subheader(
            "📊 Ranking Summary"
        )


        display_results = top_5[

            [
                "Career",
                "Final_Score"
            ]

        ].copy()


        display_results.insert(

            0,

            "Rank",

            range(
                1,
                len(display_results) + 1
            )

        )


        display_results["Final_Score"] = (

            display_results["Final_Score"]
            * 100

        ).round(2)


        display_results = (
            display_results
            .rename(
                columns={
                    "Final_Score":
                    "Suitability (%)"
                }
            )
        )


        st.dataframe(

            display_results,

            use_container_width=True,

            hide_index=True

        )


        # ====================================================
        # DEBUG / INFORMATION
        # ====================================================

        with st.expander(
            "🔍 See Recommendation Details"
        ):

            st.write(
                "Selected Field:",
                field
            )

            st.write(
                "Number of careers evaluated:",
                len(results)
            )

            st.write(
                "NOT_RECOMMENDED removed:",
                "Yes"
            )


    except Exception as e:

        st.error(
            "❌ Prediction failed."
        )

        st.code(
            str(e)
        )
