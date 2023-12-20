import os
import torch
import numpy as np
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit import RDLogger
from Levenshtein import distance as lev
from rdkit.Chem import RDKFingerprint
from transformers import BertTokenizerFast, BertForPreTraining
from loguru import logger
from fcd import get_fcd, load_ref_model, canonical_smiles
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

from utils import Recorder

RDLogger.DisableLog('rdApp.*')

# All evaluation code comes from existing work, we only modified the interface to fit our code.
# See details in Appendix C.

@torch.no_grad()
class MolT5Evaluator():
    def __init__(self, args, tokenizer):
        os.environ["TOKENIZERS_PARALLELISM"] = "True"
        self.pred_list = []
        self.label_list = []
        self.tokenizer = tokenizer
        logger.info('MolT5 evaluator initialized')

    def __call__(self, logits, labels, no_decode=False):
        if no_decode:
            self.pred_list.extend(logits)
            self.label_list.extend(labels)
            return

        if logits[0].dtype != torch.long:
            logits = logits.argmax(dim=-1)
            logits = torch.unbind(logits, dim=0)

        labels[labels == -100] = 0
        labels = torch.unbind(labels, dim=0)

        preds = [self.tokenizer.decode(logit, skip_special_tokens=True) for logit in logits]
        text_labels = [[self.tokenizer.decode(label, skip_special_tokens=True)] for label in labels]

        del logits
        del labels
        self.pred_list.extend(preds)
        self.label_list.extend(text_labels)

    def evaluate(self):
        text_model = "allenai/scibert-scivocab-uncased"
        text_trunc_length = 512
        outputs = [[None, label[0], pred] for label, pred in zip(self.label_list, self.pred_list)]

        text_tokenizer = BertTokenizerFast.from_pretrained(text_model)

        bleu_scores = []
        meteor_scores = []

        references = []
        hypotheses = []

        for i, (smi, gt, out) in enumerate(outputs):

            gt_tokens = text_tokenizer.tokenize(gt, truncation=True, max_length=text_trunc_length,
                                                padding='max_length')
            gt_tokens = list(filter(('[PAD]').__ne__, gt_tokens))
            gt_tokens = list(filter(('[CLS]').__ne__, gt_tokens))
            gt_tokens = list(filter(('[SEP]').__ne__, gt_tokens))

            out_tokens = text_tokenizer.tokenize(out, truncation=True, max_length=text_trunc_length,
                                                padding='max_length')
            out_tokens = list(filter(('[PAD]').__ne__, out_tokens))
            out_tokens = list(filter(('[CLS]').__ne__, out_tokens))
            out_tokens = list(filter(('[SEP]').__ne__, out_tokens))

            references.append([gt_tokens])
            hypotheses.append(out_tokens)

            mscore = meteor_score([gt_tokens], out_tokens)
            meteor_scores.append(mscore)

        bleu2 = corpus_bleu(references, hypotheses, weights=(.5,.5))
        bleu4 = corpus_bleu(references, hypotheses, weights=(.25,.25,.25,.25))

        _meteor_score = np.mean(meteor_scores)

        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])

        rouge_scores = []

        references = []
        hypotheses = []

        for i, (smi, gt, out) in enumerate(outputs):

            rs = scorer.score(out, gt)
            rouge_scores.append(rs)

        rouge_1 = np.mean([rs['rouge1'].fmeasure for rs in rouge_scores])
        rouge_2 = np.mean([rs['rouge2'].fmeasure for rs in rouge_scores])
        rouge_l = np.mean([rs['rougeL'].fmeasure for rs in rouge_scores])

        return {
            'bleu2': bleu2,
            'bleu4': bleu4,
            'rouge1': rouge_1,
            'rouge2': rouge_2,
            'rougeL': rouge_l,
            'mentor': _meteor_score
        }
    
    def reset(self):
        del self.pred_list
        del self.label_list
        self.pred_list = []
        self.label_list = []

@torch.no_grad()
class MolT5Evaluator_cap2smi():
    def __init__(self, args, tokenizer):
        os.environ["TOKENIZERS_PARALLELISM"] = "True"
        self.args = args
        self.pred_list = []
        self.label_list = []
        self.tokenizer = tokenizer
        logger.info('MolT5_cap2smi evaluator initialized')

    def __call__(self, logits, labels, no_decode=False):
        if no_decode:
            self.pred_list.extend(logits)
            self.label_list.extend(labels)
            return

        if logits[0].dtype != torch.long:
            logits = logits.argmax(dim=-1)
            logits = torch.unbind(logits, dim=0)

        labels[labels == -100] = 0
        labels = torch.unbind(labels, dim=0)

        preds = [self.tokenizer.decode(logit, skip_special_tokens=True) for logit in logits]
        text_labels = [self.tokenizer.decode(label, skip_special_tokens=True) for label in labels]

        del logits
        del labels
        self.pred_list.extend(preds)
        self.label_list.extend(text_labels)

    def evaluate(self):
        res1 = self.evaluate_1()
        res2 = self.evaluate_2()
        res3 = self.evaluate_3()
        return {**res2, **res1, **res3}

    def evaluate_1(self):
        outputs = []
        bad_mols = 0
        for label, pred in zip(self.label_list, self.pred_list):
            try:
                gt_smi = label
                ot_smi = pred
                gt_m = Chem.MolFromSmiles(gt_smi)
                ot_m = Chem.MolFromSmiles(ot_smi)

                if ot_m == None or gt_m == None:
                    raise ValueError('Bad SMILES')
                outputs.append((gt_m, ot_m))
            except:
                bad_mols += 1
                continue

        validity_score = len(outputs)/(len(outputs)+bad_mols)
        # logger.info('validity:', validity_score)

        MACCS_sims = []
        morgan_sims = []
        RDK_sims = []

        enum_list = outputs
        morgan_r = 2

        for i, (gt_m, ot_m) in enumerate(enum_list):

            MACCS_sims.append(DataStructs.TanimotoSimilarity(MACCSkeys.GenMACCSKeys(gt_m), MACCSkeys.GenMACCSKeys(ot_m)))
            RDK_sims.append(DataStructs.TanimotoSimilarity(RDKFingerprint(gt_m), RDKFingerprint(ot_m)))
            morgan_sims.append(DataStructs.TanimotoSimilarity(AllChem.GetMorganFingerprintAsBitVect(gt_m, morgan_r), AllChem.GetMorganFingerprintAsBitVect(ot_m, morgan_r)))

        maccs_sims_score = np.mean(MACCS_sims)
        rdk_sims_score = np.mean(RDK_sims)
        morgan_sims_score = np.mean(morgan_sims)
        # logger.info('Average MACCS Similarity:', maccs_sims_score)
        # logger.info('Average RDK Similarity:', rdk_sims_score)
        # logger.info('Average Morgan Similarity:', morgan_sims_score)

        return {
            'validity_1': validity_score,
            'maccs': maccs_sims_score,
            'rdk': rdk_sims_score,
            'morgan': morgan_sims_score
        }
    
    def evaluate_2(self):
        outputs = []

        for label, pred in zip(self.label_list, self.pred_list):
            gt_smi = label
            ot_smi = pred
            outputs.append((gt_smi, ot_smi))

        references = []
        hypotheses = []
        levs = []
        num_exact = 0
        bad_mols = 0

        for i, (gt, out) in enumerate(outputs):

            gt_tokens = [c for c in gt]

            out_tokens = [c for c in out]

            references.append([gt_tokens])
            hypotheses.append(out_tokens)

            try:
                m_out = Chem.MolFromSmiles(out)
                m_gt = Chem.MolFromSmiles(gt)

                if Chem.MolToInchi(m_out) == Chem.MolToInchi(m_gt): num_exact += 1
            except:
                bad_mols += 1

            levs.append(lev(out, gt))


        # BLEU score
        bleu_score = corpus_bleu(references, hypotheses)
        # logger.info('BLEU score:', bleu_score)

        # Exact matching score
        exact_match_score = num_exact/(i+1)
        # logger.info('Exact Match:', exact_match_score)

        # Levenshtein score
        levenshtein_score = np.mean(levs)
        # logger.info('Levenshtein:', levenshtein_score)
            
        validity_score = 1 - bad_mols/len(outputs)
        # logger.info('validity:', validity_score)

        return {
            'bleu4': bleu_score,
            'exact_match': exact_match_score,
            'levenshtein': levenshtein_score,
            'validity_2': validity_score
        }

    def evaluate_3(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        
        gt_smis = []
        ot_smis = []

        for label, pred in zip(self.label_list, self.pred_list):
            gt_smi = label
            ot_smi = pred
            if len(ot_smi) == 0: ot_smi = '[]'
            gt_smis.append(gt_smi)
            ot_smis.append(ot_smi)


        model = load_ref_model()

        canon_gt_smis = [w for w in canonical_smiles(gt_smis) if w is not None]
        canon_ot_smis = [w for w in canonical_smiles(ot_smis) if w is not None]

        fcd_sim_score = get_fcd(canon_gt_smis, canon_ot_smis, model)
        # logger.info('FCD Similarity:', fcd_sim_score)

        return {
            'fcd': fcd_sim_score
        }
    
    def reset(self):
        del self.pred_list
        del self.label_list
        self.pred_list = []
        self.label_list = []