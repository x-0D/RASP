# Rapid Abstract Syntax Based Predictor (RASP)

A hybrid language model that combines the efficiency of ROSA+ with syntactic understanding through Conditional Random Fields (CRF) and spaCy dependency parsing. RASP generates text with enhanced grammatical coherence and structural awareness.

## 🧠 System Architecture

RASP integrates three key components:

1. **ROSA+ Base Model**: A lightweight, efficient n-gram based language model
2. **Syntactic Feature Extractor**: spaCy-based dependency parsing for grammatical structure
3. **CRF Layer**: Conditional Random Fields for sequence labeling with syntactic constraints

```
Input Text → spaCy Syntactic Parser → Feature Extraction → CRF Layer → ROSA+ Base Model → Output Text
```

## 🔧 How It Works

### 1. Syntactic Feature Extraction

The system first processes text using spaCy's dependency parser to extract:

- **Token-level features**: Part-of-speech tags, dependency relations, morphological features
- **Structural features**: Dependency tree structure, ancestor relationships, children dependencies
- **Contextual features**: Subtree spans, noun chunks, sentence root information

```python
class SpacySyntacticFeatureExtractor:
    def extract_token_features(self, text: str) -> List[Dict[str, any]]:
        # Extracts per-token syntactic features including dependency tree structure
        # Returns features like pos, dep, head_pos, is_root, n_lefts, n_rights, etc.
```

### 2. CRF-Enhanced Sequence Labeling

The extracted features are fed into a CRF layer that:

- Models sequential dependencies between tokens
- Incorporates syntactic constraints into predictions
- Learns transition patterns between grammatical structures

```python
class ROSAPlusCRF:
    def _token_to_crf_features(self, tokens: List[Dict], idx: int) -> Dict[str, any]:
        # Converts token features to CRF feature format with context window
        # Includes features from previous/next tokens and dependency relationships
```

### 3. Neural CRF with BiGRU

For deeper syntactic understanding, RASP includes a neural CRF implementation:

- **Bidirectional GRU**: Captures contextual information from both directions
- **Viterbi Decoding**: Finds the most likely sequence of tags
- **Transition Parameters**: Learns syntactic transition patterns

```python
class BiGRUCRF(nn.Module):
    def viterbi_decode(self, emissions: torch.Tensor) -> List[int]:
        # Viterbi decoding for best path through syntactic states
```

### 4. Hybrid Generation Process

During text generation, RASP:

1. Gets base distribution from ROSA+ model
2. Extracts syntactic features from current context
3. Applies CRF constraints to refine predictions
4. Samples next character with syntactic awareness
5. Updates context and repeats

```python
def generate_with_syntax_constraints(self, prompt: str, max_tokens: int = 200):
    # Generates text with syntactic coherence constraints
    # Balances statistical predictions with grammatical validity
```

## 🚀 Running the Notebook

To use RASP, simply run all cells of the `ast_crf_rosa.ipynb` notebook in order:

1. The first cell installs all required dependencies including spaCy, sklearn-crfsuite, torch, and numpy
2. The second cell imports necessary libraries and modules
3. Subsequent cells define the core components of the RASP system
4. The final cells demonstrate training and usage of the complete hybrid system

The notebook contains a complete implementation with examples that will:
- Initialize the RASP model
- Train it on sample text data
- Generate text with syntactic constraints
- Analyze the syntactic features of generated text

## 🎯 Key Features

- **Syntactic Coherence**: Generates text with proper grammatical structure
- **Efficient Inference**: Combines lightweight ROSA+ with targeted syntactic processing
- **Deep Understanding**: Neural CRF captures complex syntactic patterns
- **Flexible Integration**: Can be combined with other language models
- **Interpretable Output**: Provides detailed syntactic analysis of generated text

## 🤝 Acknowledgments

Special thanks to:
- [ROSA+](https://github.com/bcml-ai/rosa-plus) for the efficient base language model
- [RWKV-LM](https://github.com/BlinkDL/RWKV-LM) for architectural inspiration

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=x-0D/RASP&type=Date)](https://star-history.com/#x-0D/RASP&Date)
