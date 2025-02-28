# DALI_MATS
Analyzing the effect of cross lingual alignment in LLM-s

# Discriminative Alignment Index 

## Folder structure
Since not all folders are relevant to this project (and will be refactored shortly),  I have highlighted the pertinent files/folders below. 
### src/
- embed_extractor.py: Gets the embeddings based on the last token and the weighted embedding for each sample in a file. Outputs the embedding as a .pkl file
- compute_alignment_xstorycloze.py : Calculates the DALI for each xstorycloze sample based on the embeddings file
- compute_alignment_xcopa.py: Calculates the DALI for each xcopa sample based on the embeddings file
- compute_alignment_belebele.py: Calculates the DALI for each belebele sample based on the embeddings file

### data
Has the necessary .txt files that are used to extract embeddings

- xcopa_dali/: Has 500 sentences (2 options) for 10 languages including the corresponding English translation
- xstorycloze_dali: Has 1511 sentences (2 options) for 10 languages + English

### alignment_outputs/
- Has the necessary DALI files (as .json) by layer for each sample
- Each dali file is structured to be {Layer_id: {sample_id: 1/0}}

### Notebook files
- Has the xstorycloze and xcopa plotting file
- Can be combined into a single notebook later

## Concept
- Conceptually, DALI is simple. For any multilingual discriminative benchmarks (with multiple labels) that are parallel (ie., same sentences/paragraphs across all languages), we compute the embeddings associated with each option. For the story completion benchmark with two labels (ending 1 or ending 2), we compute the embeddings of story+ending1 and story+ending2 for each sample across all languages.
- Compute the cosine similarity(XX,En) for sample embeddings between a non-English and the corresponding English sample across the layer of the transformer.
- DALI=1 if CS of matched pairs is higher than mismatched pairs. The model can align the two options structurally in a non-English language as it does in English. DALI=0 otherwise. 

## Accuracy
- I used lm-harness, an evaluation suite built by Eleuther AI, to evaluate the LLM for these tasks. 
- I haven't changed any of the default settings of their evaluation suite while conducting these evaluations, although I toyed with fewshot (0 shot vs 5 shot experiments) and length normalization settings

 
