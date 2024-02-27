import streamlit as st

# Page layout
st.set_page_config(layout="wide") # check this out, if i can change it.




# Function to display profile information
def display_profile():
    st.markdown("<h1 style='text-align: center;'>Esmail Atta Gumaan</h1>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style='display: flex; justify-content: center;'>
            <div style='margin: 0px 15px;'>
                <a href="https://github.com/Esmail-ibraheem" target="_blank"><img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" width="30"></a>
            </div>
            <div style='margin: 0px 15px;'>
                <a href="https://www.linkedin.com/in/esmail-a-gumaan/overlay/about-this-profile/?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base%3Bq0oKg70tTTWC9g5ncLpjSQ%3D%3D" target="_blank"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/ca/LinkedIn_logo_initials.png/768px-LinkedIn_logo_initials.png" width="30"></a>
            </div>
            <div style='margin: 0px 15px;'>
               <a href="esm.agumaan@gmail.com"><img src="https://th.bing.com/th/id/OIP.s7EtuTVtm1-iBfs188J-lAAAAA?w=474&h=474&rs=1&pid=ImgDetMain" width="30"></a>
        </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.write("# Personal Information")
    st.write("- **Degree:** B.A. Computer science student")
    st.write("- **Univeristy:** Sana'a University")
    st.write("- **Location:** Yemen/Sana'a")
    st.write("---")
    st.write("## About Me")
    st.write("B.A. Computer science student, AI engineer, Passionate about deep learning, and neural networks, keen to make things from scratch")
    st.write("---")

# Function to display projects
def display_projects():
    st.write("## Projects")
    
    # Paper Implementations Subsection
    st.write("### Paper Implementations")
    st.write("#### Transformer model")
    st.write("**Title:** Attention is All you need")
    st.write("**Description:**I built the Transformer model itself from scratch from the paper ""Attention is all you need"", Feel free to use this model for your specific purposes: translation, text generation, etc... ")
    st.latex("P E(pos,2i) = sin(pos/100002i/dmodel )")
    st.write("**GitHub:** [Link to GitHub Repository](https://github.com/Esmail-ibraheem/Transformer-model)")
    st.info("""
1. Input Embeddings: 
    The input sequence is transformed into fixed-dimensional embeddings, typically composed of word embeddings and positional encodings. 
    Word embeddings capture the semantic meaning of each word.
2. Encoder and Decoder: 
    The Transformer model consists of an encoder and a decoder. 
    Both the encoder and decoder are composed of multiple layers. Each layer has two sub-layers: 
    a multi-head self-attention mechanism and a feed-forward neural network. 
    - Encoder: 
        The encoder takes the input sequence and processes it through multiple layers of self-attention and feed-forward networks. 
        It captures the contextual information of each word based on the entire sequence. 
    - Decoder: 
        The decoder generates the output sequence word by word, attending to the encoded input sequence's relevant parts. 
        It also includes an additional attention mechanism called "encoder-decoder attention" that helps the model focus on the input during decoding.
3. Self-Attention Mechanism: 
    First, what is self-attention? 
        It is the core of the Transformer model is the self-attention mechanism. 
        It allows each word in the input sequence to attend to all other words, capturing their relevance and influence, 
        works by seeing how similar and important each word is to all of the words in a sentence, including itself. 
    Second, the Mechanism: 
        - Multi-head attention in the encoder block: 
            Plays a crucial role in capturing different types of information and learning diverse relationships between words. 
            It allows the model to attend to different parts of the input sequence simultaneously and learn multiple representations of the same input. 
        - Masked Multi-head attention in the decoder block: 
            The same as Multi-head attention in the encoder block but this time for the translation sentence, 
            is used to ensure that during the decoding process, each word can only attend to the words before it. 
            This masking prevents the model from accessing future information, which is crucial for generating the output sequence step by step. 
    - Self-attention mechanism: 
        The core of the Transformer model is the self-attention mechanism. 
        It allows each word in the input sequence to attend to all other words, capturing their relevance and influence. 
        Self-attention computes three vectors for each word: Query, Key, and Value.
        **Multi-head attention in the decoder block:** 
        Do the same as the Multi- head attention in the encoder block but between the input sentence and the translation sentence, 
        is employed to capture different relationships between the input sequence and the generated output sequence. 
        It allows the decoder to attend to different parts of the encoder's output and learn multiple representations of the context.
4. Feed Forward in two blocks: 
    It is just feed forward neural network but in this paper the neurons are 2048.
5. Add & Normalization

Self-Attention Mechanism:
        The core of the Transformer model is the self-attention mechanism. 
        It allows each word in the input sequence to attend to all other words, capturing their relevance and influence. 
        Self-attention computes three vectors for each word: Query (Q), Key (K), and Value (V).
            - Query (Q): 
                Each word serves as a query to compute the attention scores.
                Q: what I am looking for.
            - Key (K): 
                Each word acts as a key to determine its relevance to other words.
                K: what I can offer.
            - Value (V): 
                Each word contributes as a value to the attention-weighted sum.
.
""")

    code = '''
    from dataclasses import dataclass
import math
import torch 
import torch.nn as nn 
from torch.nn.functional import F 

@dataclass
class Arguments:
    source_vocab_size: int 
    target_vocab_size: int 
    source_sequence_length: int 
    target_sequence_length: int 
    d_model: int = 512 
    Layers: int = 6 
    heads: int = 8 
    dropout: float = 0.1 
    d_ff: int = 2048 

class InputEmbeddingsLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model 
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
    
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncodingLayer(nn.Module):
    def __init__(self, d_model: int, sequence_length: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.sequence_length = sequence_length
        self.dropout = nn.Dropout(dropout)

        PE = torch.zeros(sequence_length, d_model)
        Position = torch.arange(0, sequence_length, dtype=torch.float).unsqueeze(1)
        deviation_term = torch.exp(torch.arange(0, d_model, 2).float * (-math.log(10000.0) / d_model))

        PE[:, 0::2] = torch.sin(Position * deviation_term)
        PE[:, 1::2] = torch.cos(Position * deviation_term)
        PE = PE.unsqueeze(0)
        self.register_buffer('PE', PE)
    
    def forward(self, x):
        x = x + (self.PE[:, :x.shape[1], :]).requires_grad(False)
        return self.dropout(x)

    '''

    st.code(code, language='python')
    st.write("---")

    st.write("#### Paper 2")
    st.write("**Title:** Paper Title 2")
    st.write("**Description:** Brief description of the paper implementation.")
    st.write("**GitHub:** [Link to GitHub Repository](link_to_repo)")
    st.write("---")
    
    # Other Projects Subsection
    st.write("### Other Projects")
    st.write("#### Project 1")
    st.write("**Title:** Project Title 1")
    st.write("**Description:** Brief description of the project.")
    st.write("**Technologies Used:** Technologies used in the project.")
    st.write("**GitHub:** [Link to GitHub Repository](link_to_repo)")
    st.write("---")

    st.write("#### Project 2")
    st.write("**Title:** Project Title 2")
    st.write("**Description:** Brief description of the project.")
    st.write("**Technologies Used:** Technologies used in the project.")
    st.write("**GitHub:** [Link to GitHub Repository](link_to_repo)")
    st.write("[Download](https://github.com/Esmail-ibraheem/Transformer-model.git)")
    st.write("---")

    # Add more projects or paper implementations as needed

# Sidebar
st.sidebar.title("Navigation")
navigation = st.sidebar.radio("Go to", ("Profile", "Projects"))

# Main content
if navigation == "Profile":
    display_profile()
elif navigation == "Projects":
    display_projects()
