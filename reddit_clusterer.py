import streamlit as st
import openai
import praw
import textwrap
import re
import pandas as pd
import io
import requests
from streamlit_lottie import st_lottie

# --- CONFIGURATIE ---
st.set_page_config(page_title="Reddit Recap", page_icon="🧠", layout="centered")

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_chat = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_x62chJ.json")

# --- TITEL & UITLEG ---
st.title("🧠 Reddit Recap")
st.markdown("### Instantly grasp the vibe of any Reddit thread.")
st_lottie(lottie_chat, height=200, key="intro")

st.markdown("""
Welcome to **Reddit Recap** — your AI-powered assistant that scans Reddit comment sections  
and organizes hundreds of opinions into **clear themes** with **detailed descriptions** and **real, full examples**.

Just paste a Reddit post link and get a structured summary of what everyone’s talking about.
""")

# --- AUTHENTICATIE ---
reddit = praw.Reddit(
    client_id=st.secrets["REDDIT_CLIENT_ID"],
    client_secret=st.secrets["REDDIT_CLIENT_SECRET"],
    username=st.secrets["REDDIT_USERNAME"],
    password=st.secrets["REDDIT_PASSWORD"],
    user_agent=st.secrets["REDDIT_USER_AGENT"]
)

openai.api_key = st.secrets["OPENAI_API_KEY"]

# --- HULPFUNCTIES ---
def extract_post_id(text):
    match = re.search(r"comments/([a-z0-9]{6,})", text)
    if match:
        return match.group(1)
    return text.strip()

def fetch_comments(post_input):
    try:
        post_id = extract_post_id(post_input)
        submission = reddit.submission(id=post_id)
        submission.comments.replace_more(limit=None)
        comments = [comment.body for comment in submission.comments.list() if comment.body]
        return comments
    except Exception as e:
        return f"Fout bij ophalen van post {post_input}: {str(e)}"

def batch_comments(comments, batch_size=50):
    for i in range(0, len(comments), batch_size):
        yield comments[i:i + batch_size]

def cluster_comments_with_openai(comments, topic_description):
    all_outputs = []
    for batch in batch_comments(comments):
        batch_text = "\n".join(batch)

        prompt = f"""
Je krijgt een lijst Reddit-commentaren over het onderwerp: **{topic_description}**.

- Groepeer deze reacties in duidelijke categorieën.
- Geef elke categorie een **korte, duidelijke naam**.
- Voeg een **beschrijving toe die uitlegt waar de reacties in die categorie over gaan.**
- Voeg vervolgens de **duidelijkste, volledige voorbeelden** toe (lange zinnen zijn prima).
- Gebruik dit exacte formaat per categorie:

# Category: [Naam]
Description: [Beschrijving]
Examples:
- voorbeeld 1
- voorbeeld 2
- voorbeeld 3
... (enzovoort indien nodig)
"""

        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Je bent een analytische AI-categorisatietool."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        all_outputs.append(response.choices[0].message.content)

    final_prompt = f"""
Je krijgt hieronder meerdere AI-clusterresultaten, afkomstig van verschillende batches reacties over het onderwerp: {topic_description}.
Vat ze samen tot een finale set categorieën, vermijd dubbelingen en zorg voor consistente structuur.

Gebruik dit formaat:
# Category: [Naam]
Description: [Beschrijving]
Examples:
- voorbeeld 1
- voorbeeld 2
- voorbeeld 3
... (enzovoort indien nodig)

Clusters:
{textwrap.dedent('\n\n'.join(all_outputs))}
"""

    final_response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Je bent een slimme categorisatie-assistent."},
            {"role": "user", "content": final_prompt}
        ],
        temperature=0.3
    )
    return final_response.choices[0].message.content

def parse_output_to_csv(output_text):
    rows = []
    current = {}
    example_count = 1

    lines = output_text.strip().splitlines()

    for line in lines:
        if line.startswith("# Category:") or line.startswith("# Categorie:"):
            if current:
                rows.append(current)
            current = {
                "Category": line.split(":", 1)[1].strip(),
                "Description": ""
            }
            example_count = 1
        elif line.lower().startswith("description") or line.lower().startswith("beschrijving"):
            current["Description"] = line.split(":", 1)[1].strip()
        elif line.strip().startswith("-"):
            current[f"Example {example_count}"] = line.strip("- ").strip()
            example_count += 1

    if current:
        rows.append(current)

    # Zorg voor consistente kolommen gebaseerd op alle categorieën
    example_keys = [k for row in rows for k in row if k.startswith("Example")]
    max_examples = max([int(k.split()[-1]) for k in example_keys]) if example_keys else 0
    for row in rows:
        for i in range(1, max_examples + 1):
            row.setdefault(f"Example {i}", "")

    return pd.DataFrame(rows)

# --- UI INPUTVELDEN ---
st.markdown("### 🔗 Paste one or more Reddit links or IDs (comma-separated):")
post_input = st.text_input("Example: https://www.reddit.com/r/Belgium2/comments/1hzjuqd/...")

st.markdown("### 🧠 What are the comments about?")
topic_input = st.text_input("e.g. How people use AI", value="How people use AI")

# --- ACTIEKNOP ---
if st.button("Start analysis 🚀"):
    raw_inputs = [p.strip() for p in post_input.split(",") if p.strip()]
    all_comments = []
    for raw in raw_inputs:
        result = fetch_comments(raw)
        if isinstance(result, list):
            all_comments.extend(result)
        else:
            st.error(result)

    if all_comments:
        st.success(f"{len(all_comments)} comments fetched. Clustering in progress... 🧠")
        with st.spinner("AI is working..."):
            output = cluster_comments_with_openai(all_comments, topic_input)
            st.markdown("### 🗂️ Recap")
            st.markdown(output)

            # --- EXPORT KNOP ---
            df = parse_output_to_csv(output)
            st.markdown("### 📅 Download as CSV")
            csv_data = df.to_csv(index=False, sep=';').encode("utf-8")
            st.download_button(
                label="📄 Download CSV",
                data=csv_data,
                file_name="reddit_recap.csv",
                mime="text/csv"
            )
    else:
        st.warning("No valid comments found. Please check your input.")
