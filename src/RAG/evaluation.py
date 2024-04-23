from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge
from konlpy.tag import Mecab

# Initialization of tokenizer and models
mecab_tokenizer = Mecab()

def evaluate_text_similarity(reference, hypothesis):
    """
    Evaluates text similarity between a reference (ground truth) and a hypothesis (model's prediction).
    
    This function calculates three key similarity metrics: BLEU-1, ROUGE-L F1, and METEOR.
    Each of these metrics assesses different aspects of text similarity based on tokenization and scoring approaches.
    
    Parameters:
    - reference (str): The reference text, which is the ground truth or the correct answer.
    - hypothesis (str): The hypothesis text, which is the prediction made by the model.
    
    Returns:
    - dict: A dictionary containing the calculated scores for BLEU-1, ROUGE-L F1, and METEOR.
    """
    # Calculate BLEU-1 Score
    bleu_1 = sentence_bleu([reference.split()], hypothesis.split(), weights=(1.0, 0, 0, 0))
    
    # Calculate ROUGE Score
    rouge = Rouge()
    rouge_scores = rouge.get_scores(hypothesis, reference)
    rouge_l_f1 = rouge_scores[0]['rouge-l']['f']
    
    # Calculate METEOR Score using tokenized words
    ref_tokens = mecab_tokenizer.morphs(reference)
    hyp_tokens = mecab_tokenizer.morphs(hypothesis)
    meteor = meteor_score([ref_tokens], hyp_tokens)
    
    return {
        'BLEU-1': bleu_1,
        'ROUGE-L F1': rouge_l_f1,
        'METEOR': meteor
    }