from nltk.translate.bleu_score import sentence_bleu

reference = list('她的故事在法国遥远的西部山上')
hypothesis = list('她的故事在法国的遥远山')
score = sentence_bleu([reference], hypothesis)
print(score)
