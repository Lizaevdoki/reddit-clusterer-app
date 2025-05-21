import streamlit as st
import praw
import openai
import textwrap
import re

# --- AUTHENTICATIE ---
reddit = praw.Reddit(
    client_id=st.secrets["REDDIT_CLIENT_ID"],
    client_secret=st.secrets["REDDIT_CLIENT_SECRET"],
    username=st.secrets["REDDIT_USERNAME"],
    password=st.secrets["REDDIT_PASSWORD"],
    user_agent=st.secrets["REDDIT_USER_AGENT"]
)

openai.api_key = st.secrets["OPENAI_API_KEY"]

# --- FUNCTIES ---
def extract_post_id(text):
    """Haal de post-ID uit een Reddit-link of geef gewoon de ID terug."""
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

- Groepeer deze in duidelijke categorieën.
- Geef elke categorie een korte naam en beschrijving.
- Voeg 2–3 concrete voorbeelden toe per categorie.
- Voeg een aparte categorie toe voor opvallend of vreemd gebruik.

Commentaren:
{batch_text}

Output als:
# Categorie: naam
Beschrijving: uitleg
Voorbeelden:
- voorbeeld
- voorbeeld
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
Vat ze samen tot een finale set categorieën, vermijd dubbelingen en zorg voor consistente structuur:

Clusters:
{textwrap.dedent('\n\n'.join(all_outputs))}

Finale output:
# Categorie: naam
Beschrijving: uitleg
Voorbeelden:
- voorbeeld
- voorbeeld
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

# --- STREAMLIT UI ---
st.set_page_config(page_title="Reddit Use Case Clusterer", page_icon="🤖")
st.title("🤖 Reddit Use Case Clusterer")

st.markdown("Plak hieronder één of meerdere Reddit post-links of ID's, gescheiden door komma’s.")
st.markdown("**Voorbeeld:** `https://www.reddit.com/r/Belgium2/comments/1hzjuqd/waarvoor_gebruiken_jullie_chat_gpt/`")

post_input = st.text_input("Reddit links of ID's")
topic_input = st.text_input("Waarover gaan de reacties?", value="hoe mensen AI gebruiken")

if st.button("Start analyse"):
    raw_inputs = [p.strip() for p in post_input.split(",") if p.strip()]
    all_comments = []
    for raw in raw_inputs:
        result = fetch_comments(raw)
        if isinstance(result, list):
            all_comments.extend(result)
        else:
            st.error(result)

    if all_comments:
        st.success(f"{len(all_comments)} reacties opgehaald.")
        with st.spinner("AI is aan het clusteren..."):
            output = cluster_comments_with_openai(all_comments, topic_input)
            st.markdown("---")
            st.markdown(output)
    else:
        st.warning("Geen geldige reacties gevonden. Check je input.")
