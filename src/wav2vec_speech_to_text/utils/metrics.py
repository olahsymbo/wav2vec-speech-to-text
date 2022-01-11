import jiwer

def compute_metrics(preds, refs):
    """
    Compute Word Error Rate (WER) and Character Error Rate (CER) between predictions and references.
    Args:
        preds (list of str): List of predicted strings.
        refs (list of str): List of reference strings. 
    Returns:
        tuple: (WER, CER) where WER is the word error rate and CER is the character error rate.
    """
    
    wer = jiwer.wer(refs, preds)
    cer = jiwer.cer(refs, preds)
    return wer, cer
