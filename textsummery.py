import torch
import gradio as gr
from transformers import pipeline

# Load model

model_path = "sshleifer/distilbart-cnn-12-6"
text_summery = pipeline("summarization", model=model_path)

text = ("Donald John Trump was born on June 14, 1946, "
        "at Jamaica Hospital in the New York City borough of Queens,"
        " the fourth child of Fred Trump and Mary Anne MacLeod Trump.[1]"
        " He is of German and Scottish descent.[2] He grew up with his "
        "older siblings, Maryanne, Fred Jr, and Elizabeth, and his younger "
        "brother, Robert, in a 23-room mansion in the Jamaica Estates neighborhood of Queens."
        "During his first presidency, Trump imposed a travel ban on seven Muslim-majority"
        " countries, expanded the Mexico–United States border wall, and enforced a "
        "family separation policy on the border. He rolled back environmental and "
        "business regulations, signed the Tax Cuts and Jobs Act, and appointed "
        "three Supreme Court justices. He withdrew the U.S. from agreements on climate"
        " and trade and started a trade war with China. In response to the COVID-19"
        " pandemic in 2020, he downplayed its severity, contradicted health officials,"
        " and signed the bipartisan CARES Act.")

def summarize_text(text):
   result = text_summery(text)
   return (result[0]["summary_text"])

demo = gr.Interface( fn=summarize_text, inputs=gr.Textbox(lines=8, label="Enter your text"), outputs=gr.Textbox(label="Your summary"), title="AI Text Summarizer", description="Summarize long text into shorter text." )
demo.launch()

#run your code this way....
#go to genAi-project terminal
# type this Deactiavte (to deactiavte your current version of venv file)
#.\.venv311\Scripts\Activate.ps1
# python .\text-summarization\textsummery.py
#open local host from here in venv311 version