import os
import random
import streamlit as st
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import google.generativeai as genai
from newspaper import Article
from PIL import Image, ImageDraw
from io import BytesIO
from transformers import pipeline

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY", "AIzaSyBXdCeIOxwHFY31kK98htmAmU5TvR6J1zg"))

# Configure generation parameters
generation_config = {
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 40,
    "max_output_tokens": 1024,
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

# Initialize image classifier
image_classifier = pipeline("image-classification", model="google/vit-base-patch16-224")

# Scrape content and images dynamically
def scrape_content_and_images(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, "html.parser")
        text = " ".join([p.text for p in soup.find_all("p")])
        images = [img["src"] for img in soup.find_all("img") if "src" in img.attrs]
        return text, images
    except Exception as e:
        return f"Error extracting content: {e}", []

# Evaluate content with Gemini API
def evaluate_content_with_gemini(content, sensitivity, filters):
    """
    Evaluates content for safety using filters and sensitivity level.
    Provides feedback on why content is flagged as safe or unsafe.
    """
    try:
        if not content.strip():
            return {
                "status": "Error",
                "details": "Scraped content is empty. Cannot evaluate.",
                "recommendation": "Please check the website URL or try a different site.",
            }

        filters_text = ", ".join(filters) if filters else "Violence, Profanity, Explicit Language, Politics, Hate Speech"
        prompt = (
            f"Evaluate the following content to determine if it is appropriate for children based on a sensitivity level of {sensitivity}. "
            f"Consider if it contain content related to following filters: {filters_text}. Provide detailed feedback explaining why the content "
            f"is flagged as either 'Safe' or 'Unsafe'.\n\nContent:\n{content[:2000]}"
        )
        chat_session = model.start_chat(
            history=[
                {"role": "user", "parts": [prompt]}
            ]
        )
        
        response = chat_session.send_message("Please evaluate the content.")
        
        # Debugging: Print the raw response for verification
        print("Gemini Response:", response.text)
        chat_session.history = []  

        # Improved keyword matching
        if "unsafe" in response.text.lower():
                    return {
                        "status": "Unsafe",
                        "details": response.text,
                        "recommendation": "Content is flagged as inappropriate. Blocking recommended.",
                    }
        elif "safe" in response.text.lower():
            return {
                "status": "Safe",
                "details": response.text,
                "recommendation": "Content is appropriate for children.",
            }     
        else:
                return {
                "status": "Unclear",
                "details": response.text,
                "recommendation": "Content evaluation is inconclusive. Manual review recommended.",
            }
    except Exception as e:
        return {
            "status": "Error",
            "details": str(e),
            "recommendation": "Unable to evaluate content.",
        }

# Analyze images
def evaluate_image_labels_with_gemini(labels, image_sensitivity, filters=None, img_url=None):
    """
    Determines if an image is safe based on its labels and sensitivity using Gemini.
    Optionally provides the image URL for additional context and resets session history.
    """
    if filters is None:
        filters = ["Violence", "Profanity", "Explicit Content", "Politics", "Hate Speech"]

    try:
        # Construct the evaluation prompt
        prompt = (
            f"Evaluate the following image labels to determine if the content is appropriate for children based on a sensitivity level of {image_sensitivity}. "
            f"Consider if it contains content related to the following filters: {filters}. Use the labels to infer the image's context and potential risks to children. "
            f"Provide concise feedback explaining why the content is flagged as 'Safe' or 'Unsafe'.\n\nLabels:\n{labels}"
        )
        if img_url:
            prompt += f"\nThe image can be found at: {img_url}"

        # Start a fresh chat session
        chat_session = model.start_chat(history=[])

        # Validate the chat session
        if not chat_session or not hasattr(chat_session, "send_message"):
            raise ValueError("Failed to initialize chat session with Gemini model.")

        # Send prompt for evaluation
        response = chat_session.send_message(prompt)

        # Validate response
        if not response or not hasattr(response, "text") or response.text is None:
            raise ValueError("No valid response received from Gemini API.")

        feedback = response.text.lower()

        # Debugging logs
        print(f"Feedback: {feedback}")

        # Prioritize 'Unsafe' feedback over 'Safe'
        if "unsafe" in feedback:
            return "Unsafe", response.text
        elif "safe" in feedback:
            return "Safe", response.text
        else:
            return "Unclear", "The evaluation did not return a clear 'Safe' or 'Unsafe' determination."
    except Exception as e:
        return "Error", str(e)

# Adjust the analyze_images function to pass the image URL to the evaluation
def analyze_images(image_urls, base_url, preferences, image_sensitivity):
    """
    Analyzes and processes images for safety and appropriateness.
    """
    original_images = []
    processed_images = []

    for img_url in image_urls:
        if not img_url.startswith("http"):
            img_url = requests.compat.urljoin(base_url, img_url)
        try:
            # Fetch the image
            response = requests.get(img_url, timeout=10)
            if response.status_code != 200:
                raise ValueError(f"Failed to fetch image. HTTP Status Code: {response.status_code}")

            image = Image.open(BytesIO(response.content)).convert("RGB")

            # Perform image classification
            analysis = image_classifier(image)
            labels = ", ".join([label["label"] for label in analysis]) if analysis else "Unknown"
            safety_status, feedback = evaluate_image_labels_with_gemini(
                labels, image_sensitivity, preferences["filters"], img_url
            )

            # Debugging logs
            print(f"Image URL: {img_url}")
            print(f"Labels: {labels}")
            print(f"Safety Status: {safety_status}, Feedback: {feedback}")

            # Append original image details
            original_images.append((img_url, image, analysis))

            # Apply mitigation based on safety status
            if safety_status == "Unsafe":
                placeholder = create_placeholder_image("Blocked Content")
                processed_images.append((img_url, placeholder, f"Blocked: {feedback}"))
            elif safety_status == "Safe":
                processed_images.append((img_url, image, f"Safe: {feedback}"))
            else:  # Handle "Unclear" or "Error" statuses
                placeholder = create_placeholder_image("Unclear Content")
                processed_images.append((img_url, placeholder, f"Unclear: {feedback}"))

        except Exception as e:
            print(f"Error processing image {img_url}: {e}")
            placeholder = create_placeholder_image("Error Loading Image")
            original_images.append((img_url, placeholder, None))
            processed_images.append((img_url, placeholder, f"Error: {e}"))

    return original_images, processed_images

# Placeholder image for blocked content
def create_placeholder_image(text="Blocked Content"):
    img = Image.new("RGB", (224, 224), color=(255, 0, 0))
    draw = ImageDraw.Draw(img)
    text_bbox = draw.textbbox((0, 0), text)
    x = (img.width - (text_bbox[2] - text_bbox[0])) // 2
    y = (img.height - (text_bbox[3] - text_bbox[1])) // 2
    draw.text((x, y), text, fill=(255, 255, 255))
    return img


# Generate clean content
def generate_clean_content(content):
    try:
        chat_session = model.start_chat(
            history=[
                {
                    "role": "user",
                    "parts": [
                        f"Rewrite the following content to make it suitable for children as a positive story show wisdoms: {content[:1000]}"
                    ],
                }
            ]
        )
        response = chat_session.send_message("Rewrite this content.")
        chat_session.history = []  

        return response.text
    except Exception as e:
        return f"Error generating clean content: {e}"

# Generate general educational content
def generate_educational_content():
    """
    Generates random and diverse educational content with different topics.
    """
    topics = ["science", "history", "art", "mathematics", "geography", "nature"]
    selected_topic = random.choice(topics)
    try:
        chat_session = model.start_chat(
            history=[
                {
                    "role": "user",
                    "parts": [
                        f"Create an engaging and educational piece suitable for children about {selected_topic}. "
                        "Include unique insights, fun facts, or inspiring ideas to make it captivating."
                    ],
                }
            ]
        )
        response = chat_session.send_message("Generate unique educational content.")
        chat_session.history = []  # Clear history after use
        return response.text
    except Exception as e:
        return f"Error generating educational content: {e}"
    
# Helper Function to Format Text
def format_text_for_display(text):
    """
    Cleans and formats text for better display, ensuring consistent style.
    """
    import re
    cleaned_text = re.sub(r"\s+", " ", text).strip()
    paragraphs = cleaned_text.split(". ")
    return "\n\n".join(paragraphs)

# Streamlit UI
st.title("WebSitter - AI-Powered Parental Control Tool")
st.write("Analyze website content and images for suitability for children.")

# Sidebar Preferences Section
st.sidebar.header("Settings")
preferences = {
    "image_analysis": st.sidebar.checkbox("Enable Image Analysis", value=True),
    "text_sensitivity": st.sidebar.slider("Set Text Sensitivity Level", 1, 5, 3, help="Set the confidence score threshold for text analysis.",
),
    "image_sensitivity": st.sidebar.slider(
    "Set Image Sensitivity Level",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05,
    help="Set the confidence score threshold for image analysis.",
),
    "filters": st.sidebar.multiselect(
        "Content Filters", ["Violence", "Profanity", "Explicit Language", "Politics", "Hate Speech"]
    ),

    "mitigation_action": st.sidebar.radio(
        "Mitigation Action",
        ["Block Content", "Produce Clean Safe Version", "Produce Alternative Educational Content"],
        index=0
    )
}

# Input Section
url = st.text_input("Enter Website URL:", placeholder="https://example.com")

if st.button("Evaluate"):
    if url:
        with st.spinner("Scraping and analyzing content..."):
            st.info("Fetching website content...")

            # Scrape content and images
            content, image_urls = scrape_content_and_images(url)

            if "Error" in content:
                st.error(content)
            else:
                st.info("Evaluating content...")

                # Define tabs dynamically
                tab_list = ["Original Content"]
                if preferences["image_analysis"]:
                    tab_list.extend(["Original Images", "Processed Images"])
                if preferences["mitigation_action"] == "Produce Clean Safe Version":
                    tab_list.append("Clean Content")
                if preferences["mitigation_action"] == "Produce Alternative Educational Content":
                    tab_list.append("Educational Content")

                # Create tabs
                tabs = st.tabs(tab_list)

                # Original Content Tab
                with tabs[0]:
                    st.markdown("## Original Content")
                    formatted_original = format_text_for_display(content)
                    st.markdown(f"{formatted_original[:1000]}")  # Display Original Content

                    # Only show the original content evaluation here
                    result = evaluate_content_with_gemini(content, preferences["text_sensitivity"], preferences["filters"])
                    if result["status"] == "Unsafe":
                        st.error(f"Content Evaluation Result: {result['status']}")
                        st.warning(f"Details: {result['details']}")
                        st.warning(result["recommendation"])
                    else:
                        st.success(f"Content Evaluation Result: {result['status']}")
                        st.info(result["recommendation"])
                # Analyze images only once using the modified analyze_images function
                original_images, processed_images = analyze_images(
    image_urls, url, preferences, preferences["image_sensitivity"]
)

                # Original Images Tab
                if preferences["image_analysis"] and "Original Images" in tab_list:
                    with tabs[tab_list.index("Original Images")]:
                        st.subheader("Original Images")
                        for img_url, img, analysis in original_images:
                            st.image(img, caption=f"Image from {img_url}", use_container_width=True)
                            st.write("**Classification Results:**")
                            for result in analysis or []:
                                if "label" in result and "score" in result:
                                    st.write(f"- **{result['label']}**: {result['score']:.2f}")
                                else:
                                    st.error("Invalid analysis result format.")

                # Processed Images Tab
                if preferences["image_analysis"] and "Processed Images" in tab_list:
                    with tabs[tab_list.index("Processed Images")]:
                        st.subheader("Processed Images")
                        for img_url, img, feedback in processed_images:
                            st.image(img, caption=f"Processed Image from {img_url}", use_container_width=True)
                            st.write("**Feedback:**")
                            st.write(feedback)
                # Clean Content Tab
                if "Clean Content" in tab_list:
                    with tabs[tab_list.index("Clean Content")]:
                        st.markdown("## Clean Content")
                        clean_content = generate_clean_content(content)
                        if clean_content.startswith("Error") or not clean_content.strip():
                            st.error("Failed to generate clean content. Please try again or check the input.")
                        else:
                            st.markdown(clean_content)

                # Educational Content Tab
                if "Educational Content" in tab_list:
                    with tabs[tab_list.index("Educational Content")]:
                        st.markdown("## Educational Content")
                        educational_content = generate_educational_content()
                        if educational_content.startswith("Error") or not educational_content.strip():
                            st.error("Failed to generate educational content. Please try again or check the input.")
                        else:
                            st.markdown(educational_content)
