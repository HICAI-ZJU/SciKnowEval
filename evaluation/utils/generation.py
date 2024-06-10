
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from tqdm import tqdm
from rouge_score import rouge_scorer

from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys, AllChem
from rdchiral.chiral import copy_chirality
from rdkit.Chem.AllChem import AssignStereochemistry


def calculate_nltk_scores(tokenizer, ans_strs, pred_strs):
    delete_id = [i for i, x in enumerate(pred_strs) if x.strip() == '']
    ans_strs = [x for i, x in enumerate(ans_strs) if i not in delete_id]
    pred_strs = [x for i, x in enumerate(pred_strs) if i not in delete_id]
    
    ans_str_tokens = [[[tokenizer.decode([s]) for s in tokenizer.encode(ans_str)[:1024]]] for ans_str in ans_strs]
    pred_str_tokens = [[tokenizer.decode([s]) for s in tokenizer.encode(pred_str)[:1024]] for pred_str in pred_strs]
    
    delete_id = [i for i in range(len(pred_str_tokens)) if len(pred_str_tokens[i]) == 0 or len(ans_str_tokens[i][0]) == 0]
    ans_str_tokens = [ans_str_tokens[i] for i in range(len(ans_str_tokens)) if i not in delete_id]
    pred_str_tokens = [pred_str_tokens[i] for i in range(len(pred_str_tokens)) if i not in delete_id]
    
    bleu_2, bleu_4, meteor, scores = [], [], [], []
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    for i, (ans_str_token, pred_str_token) in tqdm(enumerate(zip(ans_str_tokens, pred_str_tokens)), total=len(ans_str_tokens)):
        try:
            bleu_2.append(sentence_bleu(ans_str_token, pred_str_token, weights=(0.5,0.5)))
        except:
            bleu_2.append(0)
        try:
            bleu_4.append(sentence_bleu(ans_str_token, pred_str_token, weights=(0.25,0.25,0.25,0.25)))
        except:
            bleu_4.append(0)
        # meteor.append(meteor_score(ans_str_token, pred_str_token))
        
        try:
            scores.append(rouge.score(pred_strs[i], ans_strs[i]))
        except:
            continue
     
    return {
        'bleu_2': np.mean(bleu_2),
        'bleu_4': np.mean(bleu_4),
        'rouge_1': np.mean([score['rouge1'].fmeasure for score in scores]),
        'rouge_2': np.mean([score['rouge2'].fmeasure for score in scores]),
        'rouge_l': np.mean([score['rougeL'].fmeasure for score in scores]),
        # 'meteor': np.mean(meteor)
    }

# Molecule Generation
def calculate_smiles_metrics(
        preds_smiles_list, 
        golds_smiles_list,
        metrics=('exact_match', 'fingerprint')
):
    num_all = len(preds_smiles_list)
    assert num_all > 0
    assert num_all == len(golds_smiles_list)
    k = len(preds_smiles_list[0])

    dk_pred_smiles_list_dict = {}
    dk_pred_no_answer_labels_dict = {}
    dk_pred_invalid_labels_dict = {}
    for dk in range(k):
        dk_pred_smiles_list_dict[dk] = []
        dk_pred_no_answer_labels_dict[dk] = []
        dk_pred_invalid_labels_dict[dk] = []
    for pred_smiles_list in tqdm(preds_smiles_list, desc='preds_smiles_list'):
        if pred_smiles_list is None or len(pred_smiles_list) == 0:
            for dk in range(k):
                dk_pred_no_answer_labels_dict[dk].append(True)
                dk_pred_invalid_labels_dict[dk].append(False)
                dk_pred_smiles_list_dict[dk].append(None)
            continue
        assert len(pred_smiles_list) == k
        for dk, item in enumerate(pred_smiles_list):
            item = item.strip()[:2048]
            if item == '' or item is None:
                item = None
                dk_pred_no_answer_labels_dict[dk].append(True)
                dk_pred_invalid_labels_dict[dk].append(False)
            else:
                dk_pred_no_answer_labels_dict[dk].append(False)
                try:
                    item = canonicalize_molecule_smiles(item)
                except:
                    item = None
                if item is None:
                    dk_pred_invalid_labels_dict[dk].append(True)
                else:
                    dk_pred_invalid_labels_dict[dk].append(False)
            dk_pred_smiles_list_dict[dk].append(item)
    
    new_list = []
    for gold_smiles_list in tqdm(golds_smiles_list, desc='canonicalize gold smiles'):
        sample_gold_smiles_list = []
        for gold in gold_smiles_list:
            item = gold.strip()
            new_item = canonicalize_molecule_smiles(item, return_none_for_error=False)
            # if new_item is None:
            #     new_item = item  #TODO
            # assert new_item is not None, item
            sample_gold_smiles_list.append(new_item)
        new_list.append(sample_gold_smiles_list)
    golds_smiles_list = new_list

    metric_results = {'num_all': num_all}

    tk_pred_no_answer_labels = np.array([True] * num_all)
    tk_pred_invalid_labels = np.array([True] * num_all)
    for dk in range(k):
        dk_no_answer_labels = dk_pred_no_answer_labels_dict[dk]
        dk_invalid_labels = dk_pred_invalid_labels_dict[dk]
        tk_pred_no_answer_labels = tk_pred_no_answer_labels & dk_no_answer_labels
        tk_pred_invalid_labels = tk_pred_invalid_labels & dk_invalid_labels
        metric_results['num_t%d_no_answer' % (dk + 1)] = tk_pred_no_answer_labels.sum().item()
        metric_results['num_t%d_invalid' % (dk + 1)] = tk_pred_invalid_labels.sum().item()
    
    # d1_no_answer_labels = dk_pred_no_answer_labels_dict[0]
    # # print(np.array(d1_no_answer_labels).sum().item())
    # for label, item in zip(d1_no_answer_labels, preds_smiles_list):
    #     if label:
    #         print(item)

    for metric in metrics:
        if metric == 'exact_match':
            tk_exact_match_labels = np.array([False] * num_all)
            for dk in range(k):
                dk_pred_smiles_list = dk_pred_smiles_list_dict[dk]
                dk_exact_match_labels = judge_exact_match(dk_pred_smiles_list, golds_smiles_list)
                tk_exact_match_labels = tk_exact_match_labels | dk_exact_match_labels
                metric_results['num_t%d_exact_match' % (dk + 1)] = tk_exact_match_labels.sum().item()
        elif metric == 'fingerprint':
            d1_pred_mol_list = []
            gold_mols_list = []
            for pred_smiles, gold_smiles_list, no_answer, invalid in zip(dk_pred_smiles_list_dict[0], golds_smiles_list, dk_pred_no_answer_labels_dict[0], dk_pred_invalid_labels_dict[0]):
                if pred_smiles is None or pred_smiles.strip() == '' or no_answer is True or invalid is True:
                    continue
                pred_mol = Chem.MolFromSmiles(pred_smiles)
                # if pred_mol is None:  # TODO
                #     continue
                assert pred_mol is not None, pred_smiles
                gold_mol_list = []
                for gold_smiles in gold_smiles_list:
                    gold_mol = Chem.MolFromSmiles(gold_smiles)
                    # if gold_mol is None:
                    #     continue  # TODO
                    assert gold_mol is not None, gold_smiles
                    gold_mol_list.append(gold_mol)
                # if len(gold_mol_list) == 0:
                #     continue  # TODO
                d1_pred_mol_list.append(pred_mol)
                gold_mols_list.append(gold_mol_list)
            maccs_sims_score, rdk_sims_score, morgan_sims_score = calculate_fingerprint_similarity(d1_pred_mol_list, gold_mols_list)
            metric_results['t1_maccs_fps'] = maccs_sims_score
            metric_results['t1_rdk_fps'] = rdk_sims_score
            metric_results['t1_morgan_fps'] = morgan_sims_score
        elif metric == 'multiple_match':
            tk_intersection_labels = np.array([False] * num_all)
            tk_subset_labels = np.array([False] * num_all)
            for dk in range(k):
                dk_intersection_labels, dk_subset_labels = judge_multiple_match(dk_pred_smiles_list_dict[dk], golds_smiles_list)
                tk_intersection_labels = tk_intersection_labels | dk_intersection_labels
                tk_subset_labels = tk_subset_labels | dk_subset_labels
                metric_results['num_t%d_subset' % (dk + 1)] = tk_intersection_labels.sum().item()
                metric_results['num_t%d_intersection' % (dk + 1)] = tk_intersection_labels.sum().item()
        else:
            raise ValueError(metric)
    
    return metric_results

def convert_smiles_list_into_mol_list(smiles_list):
    # 将smiles列表转换为rdkit的mol列表
    mol_list = []
    no_answer_labels = []
    invalid_labels = []
    for smiles in smiles_list:
        if smiles == '':
            mol = 'NA'
            no_answer_labels.append(True)
        else:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                mol = 'INVALID'
                invalid_labels.append(True)
        mol_list.append(mol)
    
    no_answer_labels = np.array(no_answer_labels)
    invalid_labels = np.arange(invalid_labels)

    return mol_list, no_answer_labels, invalid_labels

def get_molecule_id(smiles, remove_duplicate=True):
    if remove_duplicate:
        assert ';' not in smiles
        all_inchi = set()
        for part in smiles.split('.'):
            inchi = get_molecule_id(part, remove_duplicate=False)
            all_inchi.add(inchi)
        all_inchi = tuple(sorted(all_inchi))
        return all_inchi
    else:
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToInchi(mol)

def judge_exact_match(pred_can_smiles_list, gold_can_smiles_list):
    # 判断预测的smiles是否和真实的smiles完全一致
    assert len(pred_can_smiles_list) == len(gold_can_smiles_list)
    exact_match_labels = []
    for pred_smiles, gold_smiles_list in zip(pred_can_smiles_list, gold_can_smiles_list):
        if pred_smiles is None:
            exact_match_labels.append(False)
            continue
        pred_smiles_inchi = get_molecule_id(pred_smiles)
        sample_exact_match = False
        for gold_smiles in gold_smiles_list:
            if gold_smiles is None:
                break
            gold_smiles_inchi = get_molecule_id(gold_smiles)
            if pred_smiles_inchi == gold_smiles_inchi:
                sample_exact_match = True
                break
        exact_match_labels.append(sample_exact_match)
    return np.array(exact_match_labels)

def calculate_fingerprint_similarity(pred_mol_list, gold_mols_list, morgan_r=2):
    # 计算预测分子与真实分子指纹相似度
    assert len(pred_mol_list) == len(gold_mols_list)
    MACCS_sims = []
    morgan_sims = []
    RDK_sims = []
    for pred_mol, gold_mol_list in zip(pred_mol_list, gold_mols_list):
        if pred_mol is None or type(pred_mol) == str:
            raise ValueError(type(pred_mol))
        tmp_MACCS, tmp_RDK, tmp_morgan = 0, 0, 0
        for gold_mol in gold_mol_list:
            tmp_MACCS = max(tmp_MACCS, DataStructs.FingerprintSimilarity(MACCSkeys.GenMACCSKeys(gold_mol), MACCSkeys.GenMACCSKeys(pred_mol), metric=DataStructs.TanimotoSimilarity))
            tmp_RDK = max(tmp_RDK, DataStructs.FingerprintSimilarity(Chem.RDKFingerprint(gold_mol), Chem.RDKFingerprint(pred_mol), metric=DataStructs.TanimotoSimilarity))
            tmp_morgan = max(tmp_morgan, DataStructs.TanimotoSimilarity(AllChem.GetMorganFingerprint(gold_mol,morgan_r), AllChem.GetMorganFingerprint(pred_mol, morgan_r)))
        MACCS_sims.append(tmp_MACCS)
        RDK_sims.append(tmp_RDK)
        morgan_sims.append(tmp_morgan)
    maccs_sims_score = np.mean(MACCS_sims)
    rdk_sims_score = np.mean(RDK_sims)
    morgan_sims_score = np.mean(morgan_sims)
    return maccs_sims_score, rdk_sims_score, morgan_sims_score


def judge_multiple_match(pred_can_smiles_list, golds_can_smiles_list):
    assert len(pred_can_smiles_list) == len(golds_can_smiles_list)
    subset_labels = []
    intersection_labels = []
    for pred_smiles, gold_smiles_list in zip(pred_can_smiles_list, golds_can_smiles_list):
        if pred_smiles is None:
            subset_labels.append(False)
            intersection_labels.append(False)
            continue

        pred_ele_set = set()
        for smiles in pred_smiles.split('.'):
            pred_ele_set.add(get_molecule_id(smiles, remove_duplicate=False))

        intersection_label = False
        subset_label = False
        for gold_smiles in gold_smiles_list:
            assert gold_smiles is not None
            gold_ele_set = set()
            for smiles in gold_smiles.split('.'):
                gold_ele_set.add(get_molecule_id(smiles, remove_duplicate=False))

            if len(pred_ele_set & gold_ele_set) > 0:
                intersection_label = True
                g_p = gold_ele_set - pred_ele_set
                if len(g_p) >= 0 and len(pred_ele_set - gold_ele_set) == 0:
                    subset_label = True
                    break
        intersection_labels.append(intersection_label)
        subset_labels.append(subset_label)
    
    return intersection_labels, subset_labels

def canonicalize(smiles, isomeric=False, canonical=True, kekulize=False):
    # When canonicalizing a SMILES string, we typically want to
    # run Chem.RemoveHs(mol), but this will try to kekulize the mol
    # which is not required for canonical SMILES.  Instead, we make a
    # copy of the mol retaining only the information we desire (not explicit Hs)
    # Then, we sanitize the mol without kekulization.  copy_atom and copy_edit_mol
    # Are used to create this clean copy of the mol.
    def copy_atom(atom):
        new_atom = Chem.Atom(atom.GetSymbol())
        new_atom.SetFormalCharge(atom.GetFormalCharge())
        if atom.GetIsAromatic() and atom.GetNoImplicit():
            new_atom.SetNumExplicitHs(atom.GetNumExplicitHs())
            #elif atom.GetSymbol() == 'N':
            #    print(atom.GetSymbol())
            #    print(atom.GetImplicitValence())
            #    new_atom.SetNumExplicitHs(-atom.GetImplicitValence())
            #elif atom.GetSymbol() == 'S':
            #    print(atom.GetSymbol())
            #    print(atom.GetImplicitValence())
        return new_atom

    def copy_edit_mol(mol):
        new_mol = Chem.RWMol(Chem.MolFromSmiles(''))
        for atom in mol.GetAtoms():
            new_atom = copy_atom(atom)
            new_mol.AddAtom(new_atom)
        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom().GetIdx()
            a2 = bond.GetEndAtom().GetIdx()
            bt = bond.GetBondType()
            new_mol.AddBond(a1, a2, bt)
            new_bond = new_mol.GetBondBetweenAtoms(a1, a2)
            new_bond.SetBondDir(bond.GetBondDir())
            new_bond.SetStereo(bond.GetStereo())
        for new_atom in new_mol.GetAtoms():
            atom = mol.GetAtomWithIdx(new_atom.GetIdx())
            copy_chirality(atom, new_atom)
        return new_mol

    smiles = smiles.replace(" ", "")  
    tmp = Chem.MolFromSmiles(smiles, sanitize=False)
    tmp.UpdatePropertyCache()
    new_mol = copy_edit_mol(tmp)
    #Chem.SanitizeMol(new_mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL)
    if not kekulize:
        Chem.SanitizeMol(new_mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_SETAROMATICITY | Chem.SanitizeFlags.SANITIZE_PROPERTIES | Chem.SanitizeFlags.SANITIZE_ADJUSTHS, catchErrors=True)
    else:
        Chem.SanitizeMol(new_mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE | Chem.SanitizeFlags.SANITIZE_PROPERTIES | Chem.SanitizeFlags.SANITIZE_ADJUSTHS, catchErrors=True)
   
    AssignStereochemistry(new_mol, cleanIt=False, force=True, flagPossibleStereoCenters=True)
    
    new_smiles = Chem.MolToSmiles(new_mol, isomericSmiles=isomeric, canonical=canonical)
    return new_smiles
    
def canonicalize_molecule_smiles(smiles, return_none_for_error=True, skip_mol=False, sort_things=True, isomeric=True, kekulization=True, allow_empty_part=False):
    things = smiles.split('.')
    if skip_mol:
        new_things = things
    else:
        new_things = []
        for thing in things:
            try:
                if thing == '' and not allow_empty_part:
                    raise ValueError(f'SMILES {smiles} contains empty part.')

                mol = Chem.MolFromSmiles(thing)
                if mol is None:
                    return None
                for atom in mol.GetAtoms():
                    atom.SetAtomMapNum(0)
                thing_smiles = Chem.MolToSmiles(mol, kekuleSmiles=False, isomericSmiles=isomeric)
                thing_smiles = Chem.MolFromSmiles(thing_smiles)
                thing_smiles = Chem.MolToSmiles(thing_smiles, kekuleSmiles=False, isomericSmiles=isomeric)
                thing_smiles = Chem.MolFromSmiles(thing_smiles)
                thing_smiles = Chem.MolToSmiles(thing_smiles, kekuleSmiles=False, isomericSmiles=isomeric)
                if thing_smiles is None:
                    return None
                can_in = thing_smiles
                can_out = canonicalize(thing_smiles, isomeric=isomeric)
                if can_out is None:
                    can_out = can_in
                thing_smiles = can_out
                if kekulization:
                    thing_smiles = keku_mid = Chem.MolFromSmiles(thing_smiles)
                    assert keku_mid is not None, 'Before can: %s\nAfter can: %s' % (can_in, can_out)
                    thing_smiles = Chem.MolToSmiles(thing_smiles, kekuleSmiles=True, isomericSmiles=isomeric)
            except KeyboardInterrupt:
                raise
            except:
                if return_none_for_error:
                    return None
                else:
                    raise
            new_things.append(thing_smiles)
    if sort_things:
        new_things = sorted(new_things)
    new_things = '.'.join(new_things)
    return new_things