import msgspec
from cidereval import cider, ciderD

from datasets import load_metric
import evaluate

bleu_metric = load_metric("bleu")
meteor_metric = load_metric("meteor")
rouge_metric = load_metric("rouge")


bleu = evaluate.load("bleu")
meteor = evaluate.load("meteor")
rouge = evaluate.load("rouge")

prefix = "eval_data/3.1-video"

with open(f'{prefix}/pred.json', 'r') as preds_file, open(f'{prefix}/gt.json', 'r') as gt_file:
    preds = msgspec.json.decode(preds_file.read())
    gts = msgspec.json.decode(gt_file.read())

    print(cider(predictions=preds, references=gts))

    bleu_results = bleu.compute(predictions=preds, references=gts, smooth=False)
    print(bleu_results)

    meteor_results = meteor.compute(predictions=preds, references=gts)
    print(meteor_results)

    rouge_results = rouge.compute(predictions=preds, references=gts)
    print(rouge_results)