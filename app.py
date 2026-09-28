import streamlit as st
import torch
import numpy as np
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="CLIP Image–Text Matching",
    page_icon="🖼️",
    layout="wide"
)


# ---------------------------------------------------------
# Title
# ---------------------------------------------------------

st.title("🖼️ CLIP Image–Text Matching System")

st.write(
    """
    Upload an image and enter multiple candidate descriptions.
    The CLIP model compares the image with each text description
    and identifies the most semantically related description.
    """
)


# ---------------------------------------------------------
# Load CLIP Model
# ---------------------------------------------------------

MODEL_NAME = "openai/clip-vit-base-patch32"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


@st.cache_resource
def load_model():

    processor = CLIPProcessor.from_pretrained(MODEL_NAME)

    model = CLIPModel.from_pretrained(MODEL_NAME)

    model = model.to(device)

    model.eval()

    return model, processor


with st.spinner("Loading CLIP model..."):
    model, processor = load_model()

st.success(f"CLIP model loaded successfully | Device: {device}")


# ---------------------------------------------------------
# Image Upload
# ---------------------------------------------------------

st.subheader("1. Upload an Image")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "webp"]
)


# ---------------------------------------------------------
# Candidate Text Input
# ---------------------------------------------------------

st.subheader("2. Enter Candidate Descriptions")

st.write(
    "Enter one description per line."
)

default_text = """A dog playing in the grass
A cat sitting on a chair
A car parked on the road
A group of people walking"""

candidate_text_input = st.text_area(
    "Candidate descriptions",
    value=default_text,
    height=180
)


# ---------------------------------------------------------
# Matching Function
# ---------------------------------------------------------

def match_image_with_text(image, candidate_texts):

    inputs = processor(
        text=candidate_texts,
        images=image,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    # Move tensors to the selected device
    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    # No gradient calculation is required for prediction
    with torch.inference_mode():

        outputs = model(**inputs)

        # CLIP image-text logits
        logits = outputs.logits_per_image[0]

        # Relative scores among candidate descriptions
        scores = torch.softmax(
            logits,
            dim=0
        ).cpu().numpy()

        # Image and text embeddings
        image_embeddings = outputs.image_embeds
        text_embeddings = outputs.text_embeds

        # Normalize embeddings
        normalized_image = (
            image_embeddings /
            image_embeddings.norm(
                dim=-1,
                keepdim=True
            )
        )

        normalized_text = (
            text_embeddings /
            text_embeddings.norm(
                dim=-1,
                keepdim=True
            )
        )

        # Cosine similarity
        cosine_scores = (
            normalized_image @
            normalized_text.T
        )[0]

        cosine_scores = cosine_scores.cpu().numpy()

    return scores, cosine_scores, logits.cpu().numpy()


# ---------------------------------------------------------
# Run Matching
# ---------------------------------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("3. Uploaded Image")

    st.image(
        image,
        caption="Uploaded Image",
        width=500
    )

    # Clean candidate descriptions
    candidate_texts = [
        text.strip()
        for text in candidate_text_input.split("\n")
        if text.strip()
    ]

    if len(candidate_texts) < 2:

        st.warning(
            "Please enter at least two candidate descriptions."
        )

    else:

        if st.button(
            "🔍 Find Best Matching Description",
            type="primary"
        ):

            with st.spinner(
                "Comparing image with text descriptions..."
            ):

                scores, cosine_scores, logits = (
                    match_image_with_text(
                        image,
                        candidate_texts
                    )
                )

            # -------------------------------------------------
            # Find Best Match
            # -------------------------------------------------

            best_index = int(
                np.argmax(scores)
            )

            best_text = candidate_texts[
                best_index
            ]

            best_score = scores[
                best_index
            ]

            # -------------------------------------------------
            # Display Best Match
            # -------------------------------------------------

            st.subheader("🎯 Best Match")

            st.success(
                f"**{best_text}**"
            )

            st.metric(
                "Relative Score",
                f"{best_score * 100:.2f}%"
            )

            # -------------------------------------------------
            # Results Table
            # -------------------------------------------------

            st.subheader(
                "📊 All Candidate Results"
            )

            results = []

            for i, text in enumerate(
                candidate_texts
            ):

                results.append(
                    {
                        "Rank": i + 1,
                        "Description": text,
                        "Relative Score": (
                            scores[i] * 100
                        ),
                        "Cosine Similarity": (
                            cosine_scores[i]
                        )
                    }
                )

            # Sort by score
            results = sorted(
                results,
                key=lambda x: x["Relative Score"],
                reverse=True
            )

            for rank, result in enumerate(
                results,
                start=1
            ):

                st.write(
                    f"**{rank}. {result['Description']}**"
                )

                col1, col2 = st.columns(2)

                with col1:

                    st.write(
                        f"Relative Score: "
                        f"{result['Relative Score']:.2f}%"
                    )

                with col2:

                    st.write(
                        f"Cosine Similarity: "
                        f"{result['Cosine Similarity']:.4f}"
                    )

                st.progress(
                    float(
                        result["Relative Score"] / 100
                    )
                )

            # -------------------------------------------------
            # Bar Chart
            # -------------------------------------------------

            st.subheader(
                "📈 Similarity Comparison"
            )

            chart_data = {
                text: float(score * 100)
                for text, score
                in zip(
                    candidate_texts,
                    scores
                )
            }

            st.bar_chart(
                chart_data
            )

            # -------------------------------------------------
            # Explanation
            # -------------------------------------------------

            st.subheader(
                "ℹ️ How the Result Works"
            )

            st.write(
                """
                CLIP converts the image and text descriptions
                into embeddings in the same feature space.

                It then calculates how strongly the image
                corresponds to each text description.

                The description with the highest relative score
                is selected as the best match.
                """
            )

            st.info(
                """
                Note: The percentages are relative scores among
                the candidate descriptions you provided. They
                should not be interpreted as absolute probabilities
                that the description is true.
                """
            )


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------

st.markdown("---")

st.caption(
    "Image–Text Matching using OpenAI CLIP | "
    "Hugging Face Transformers + Streamlit"
)
