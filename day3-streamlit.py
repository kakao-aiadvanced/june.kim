# $ pip install streamlit
# $ streamlit run day3-streamlit.py
import streamlit as st

from day3_utils import workflow

# Empty container to hold the chat.
placeholder = st.empty()

# Get user input
user_input = st.text_input("Enter your message")

from pprint import pprint

# Compile
app = workflow.compile()
inputs = {"question": "Who are the Bears expected to draft first in the NFL draft?"}
for output in app.stream(inputs):
    for key, value in output.items():
        pprint(f"Finished running: {key}:")
# pprint(value["generation"])
# pprint(value["urls"])

# Echo user input in placeholder
placeholder.text(value["generation"] + "\n" + '\n'.join(value["urls"]))

