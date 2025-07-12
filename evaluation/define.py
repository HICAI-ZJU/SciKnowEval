import json
import os
import time
from evaluation.metrics import *
from typing import List, Tuple, Any, Dict
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")


def get_task_func():
    try:
        return {
            ### Chemistry
            # L1
            #'molecule_name_conversion': get_score_CLS,
            #'molecular_property_prediction': get_score_CLS,
            'chemical_literature_QA': get_score_CLS,
            # L2
            'reaction_mechanism_inference': get_score_CLS,
            #'compound_identification_and_properties': get_score_CLS,
            'extract_doping': get_score_GPT4,
            'chemical_detailed_understanding': get_score_CLS,
            'chemical_text_summary': get_score_GPT4,
            'chemical_hypothesis_verification': get_score_CLS,
            #'chemical_reasoning_and_interpretation': get_score_CLS,
            # L3
            'molar_weight_calculation': get_score_CLS,
            'molecular_property_calculation': get_score_CLS,
            'molecule_structure_prediction': get_score_CLS,
            'reaction_prediction': get_score_reaction,
            'retrosynthesis': get_score_reaction,
            'balancing_chemical_equation': get_score_filling,
            #'chemical_calculation': get_score_CLS,
            # L4
            'chemical_harmful_QA': get_score_GPT4,
            'mol_toxicity_prediction': get_score_CLS,
            'chemical_laboratory_safety_test': get_score_CLS,
            # L5
            #'molecule_captioning': get_score_BLEU_ROUGE,
            #'molecule_generation': get_score_Mol_GEN,
            'chemical_procedure_generation': get_score_GPT4,
            'chemical_reagent_generation': get_score_GPT4,
            ### Biology
            # L1
            #'protein_property_identification': get_score_CLS,
            'biology_literature_QA': get_score_CLS,
            # L2
            'drug_drug_relation_extraction': get_score_RE_triplets,
            'biomedical_judgment_and_interpretation': get_score_CLS,
            'compound_disease_relation_extraction': get_score_RE_tuples,
            #'gene_disease_relation_extraction': get_score_RE_triplets,
            'biological_detailed_understanding': get_score_CLS,
            'biological_text_summary': get_score_GPT4,
            'biological_hypothesis_verification': get_score_CLS,
            #'biological_reasoning_and_interpretation': get_score_CLS,
            # L3
            'solubility_prediction': get_score_CLS,
            'beta_lactamase_activity_prediction': get_score_CLS,
            'fluorescence_prediction': get_score_CLS,
            'GB1_ftness_prediction': get_score_CLS,
            'stability_prediction': get_score_CLS,
            'Protein_Protein_Interaction': get_score_CLS,
            #'biological_calculation': get_score_CLS,
            # L4
            'biological_harmful_QA': get_score_GPT4,
            'proteotoxicity_prediction': get_score_CLS,
            'biological_laboratory_safety_test': get_score_CLS,
            # L5
            'biological_procedure_generation': get_score_GPT4,
            'biological_reagent_generation': get_score_GPT4,
            #'protein_description_generation': get_score_BLEU_ROUGE,
            #'protein_design': get_score_smith_waterman,
            #'single_cell_analysis': get_score_BLEU_ROUGE,
            ### Meterial
            # L1
            'material_literature_QA': get_score_CLS,
            # L2
            'material_hypothesis_verification': get_score_CLS,
            'material_component_extraction': get_score_GPT4,
            'material_data_extraction': get_score_CLS,
            'material_detailed_understanding': get_score_CLS,
            #'material_reasoning_and_interpretation': get_score_CLS,
            'material_text_summary': get_score_GPT4,
            # L3
            'valence_electron_difference_calculation': get_score_CLS,
            #'material_calculation': get_score_CLS,
            'lattice_volume_calculation': get_score_CLS,
            'perovskite_stability_prediction': get_score_CLS,
            'diffusion_rate_analysis': get_score_CLS,
            # L4
            'material_safety_QA': get_score_CLS,
            'material_toxicity_prediction': get_score_CLS,
            # L5
            'crystal_structure_and_composition_analysis': get_score_GPT4,
            'specified_band_gap_material_generation': get_score_GPT4,
            #'property_and_usage_analysis': get_score_GPT4,
            ### Physics
            # L1
            'physics_literature_QA': get_score_CLS,
            #'fundamental_physics_exam': get_score_CLS,
            # L2
            'physics_hypothesis_verification': get_score_CLS,
            'physics_detailed_understanding': get_score_CLS,
            #'physics_reasoning_and_interpretation': get_score_CLS,
            'physics_text_summary': get_score_GPT4,
            # L3
            #'high_school_physics_calculation': get_score_CLS,
            'general_physics_calculation': get_score_CLS,
            'physics_formula_derivation': get_score_GPT4,
            # L4
            'physics_safety_QA': get_score_CLS,
            'physics_laboratory_safety_test': get_score_CLS,
            # L5
            'physics_problem_solving': get_score_GPT4,
        }
    except:
        raise NotImplementedError("task not found")

def reformat_result(result: Dict[str, Any]) -> Dict[str, Any]:
    reformatted_result = {
        'Biology': {
            'L1': {
                #'protein_property_identification': result['protein_property_identification'],
                'biology_literature_QA': result['biology_literature_QA'],
            },
            'L2': {
                'drug_drug_relation_extraction': result['drug_drug_relation_extraction'],
                'biomedical_judgment_and_interpretation': result['biomedical_judgment_and_interpretation'],
                'compound_disease_relation_extraction': result['compound_disease_relation_extraction'],
                #'gene_disease_relation_extraction': result['gene_disease_relation_extraction'],
                'biological_detailed_understanding': result['biological_detailed_understanding'],
                'biological_text_summary': result['biological_text_summary'],
                'biological_hypothesis_verification': result['biological_hypothesis_verification'],
                #'biological_reasoning_and_interpretation': result['biological_reasoning_and_interpretation'],
            },
            'L3': {
                'solubility_prediction': result['solubility_prediction'],
                'beta_lactamase_activity_prediction': result['beta_lactamase_activity_prediction'],
                'fluorescence_prediction': result['fluorescence_prediction'],
                'GB1_ftness_prediction': result['GB1_ftness_prediction'],
                'stability_prediction': result['stability_prediction'],
                'Protein_Protein_Interaction': result['Protein_Protein_Interaction'],
                #'biological_calculation': result['biological_calculation'],
            },
            'L4': {
                'biological_harmful_QA': result['biological_harmful_QA'],
                'proteotoxicity_prediction': result['proteotoxicity_prediction'],
                'biological_laboratory_safety_test': result['biological_laboratory_safety_test'],
            },
            'L5': {
                'biological_procedure_generation': result['biological_procedure_generation'],
                'biological_reagent_generation': result['biological_reagent_generation'],
                #'protein_design': result['protein_design'],
                #'single_cell_analysis': result['single_cell_analysis'],
                #'protein_description_generation': result['protein_description_generation'],
            },
        },
        'Chemistry': {
            'L1': {
                #'molecule_name_conversion': result['molecule_name_conversion'],
                #'molecular_property_prediction': result['molecular_property_prediction'],
                'chemical_literature_QA': result['chemical_literature_QA'],
                #'molecule_captioning': result['molecule_captioning'],
            },
            'L2': {
                'reaction_mechanism_inference': result['reaction_mechanism_inference'],
                #'compound_identification_and_properties': result['compound_identification_and_properties'],
                'extract_doping': result['extract_doping'],
                'chemical_detailed_understanding': result['chemical_detailed_understanding'],
                'chemical_text_summary': result['chemical_text_summary'],
                'chemical_hypothesis_verification': result['chemical_hypothesis_verification'],
                #'chemical_reasoning_and_interpretation': result['chemical_reasoning_and_interpretation'],
            },
            'L3': {
                'molar_weight_calculation': result['molar_weight_calculation'],
                'molecular_property_calculation': result['molecular_property_calculation'],
                'molecule_structure_prediction': result['molecule_structure_prediction'],
                'reaction_prediction': result['reaction_prediction'],
                'retrosynthesis': result['retrosynthesis'],
                'balancing_chemical_equation': result['balancing_chemical_equation'],
                #'chemical_calculation': result['chemical_calculation'],
            },
            'L4': {
                'chemical_harmful_QA': result['chemical_harmful_QA'],
                'mol_toxicity_prediction': result['mol_toxicity_prediction'],
                'chemical_laboratory_safety_test': result['chemical_laboratory_safety_test'],
            },
            'L5': {
                #'molecule_generation': result['molecule_generation'],
                'chemical_procedure_generation': result['chemical_procedure_generation'],
                'chemical_reagent_generation': result['chemical_reagent_generation'],
            },
        },
        'Materials': {
            'L1': {
                'material_literature_QA': result['material_literature_QA'],
            },
            'L2': {
                'material_hypothesis_verification': result['material_hypothesis_verification'],
                'material_component_extraction': result['material_component_extraction'],
                'material_data_extraction': result['material_data_extraction'],
                'material_detailed_understanding': result['material_detailed_understanding'],
                #'material_reasoning_and_interpretation': result['material_reasoning_and_interpretation'],
                'material_text_summary': result['material_text_summary'],
            },
            'L3': {
                'valence_electron_difference_calculation': result['valence_electron_difference_calculation'],
                #'material_calculation': result['material_calculation'],
                'lattice_volume_calculation': result['lattice_volume_calculation'],
                'perovskite_stability_prediction': result['perovskite_stability_prediction'],
                'diffusion_rate_analysis': result['diffusion_rate_analysis'],
            },
            'L4': {
                'material_safety_QA': result['material_safety_QA'],
                'material_toxicity_prediction': result['material_toxicity_prediction'],
            },
            'L5': {
                'crystal_structure_and_composition_analysis': result['crystal_structure_and_composition_analysis'],
                'specified_band_gap_material_generation': result['specified_band_gap_material_generation'],
                #'property_and_usage_analysis': result['property_and_usage_analysis'],
            },
        },
        'Physics': {
            'L1': {
                'physics_literature_QA': result['physics_literature_QA'],
                #'fundamental_physics_exam': result['fundamental_physics_exam'],
            },
            'L2': {
                'physics_hypothesis_verification': result['physics_hypothesis_verification'],
                'physics_detailed_understanding': result['physics_detailed_understanding'],
                #'physics_reasoning_and_interpretation': result['physics_reasoning_and_interpretation'],
                'physics_text_summary': result['physics_text_summary'],
            },
            'L3': {
                #'high_school_physics_calculation': result['high_school_physics_calculation'],
                'general_physics_calculation': result['general_physics_calculation'],
                'physics_formula_derivation': result['physics_formula_derivation'],
            },
            'L4': {
                'physics_safety_QA': result['physics_safety_QA'],
                'physics_laboratory_safety_test': result['physics_laboratory_safety_test'],
            },
            'L5': {
                'physics_problem_solving': result['physics_problem_solving'],
            },
        },
    }

    return reformatted_result

def get_task_data(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    try:
        task_data = {
            # L1
            #'molecule_name_conversion': [d for d in data if d['details']['task'] == 'molecule_name_conversion'],
            #'molecular_property_prediction': [d for d in data if d['details']['task'] == 'molecular_property_prediction' and d['details']['level'] == 'L1'],
            'chemical_literature_QA': [d for d in data if d['details']['task'] == 'literature_multi_choice_question' and d['domain'] == 'Chemistry'],
            #'molecule_captioning': [d for d in data if d['details']['task'] == 'molecule_captioning'],
            # L2
            'reaction_mechanism_inference': [d for d in data if d['details']['subtask'] == 'reaction_mechanism_inference'],
            #'compound_identification_and_properties': [d for d in data if d['details']['subtask'] == 'compound_identification_and_properties'],
            'extract_doping': [d for d in data if d['details']['subtask'] == 'extract_doping'],
            'chemical_detailed_understanding': [d for d in data if d['details']['subtask'] == 'detailed_understanding' and d['domain'] == 'Chemistry'],
            'chemical_text_summary': [d for d in data if d['details']['subtask'] == 'text_summary' and d['domain'] == 'Chemistry'],
            'chemical_hypothesis_verification': [d for d in data if d['details']['subtask'] == 'hypothesis_verification' and d['domain'] == 'Chemistry'],
            #'chemical_reasoning_and_interpretation': [d for d in data if d['details']['subtask'] == 'reasoning_and_interpretation' and d['domain'] == 'Chemistry'],
            # L3
            'molar_weight_calculation': [d for d in data if d['details']['task'] == 'molar_weight_calculation'],
            'molecular_property_calculation': [d for d in data if d['details']['task'] == 'molecular_property_prediction' and d['details']['level'] == 'L3'],
            'molecule_structure_prediction': [d for d in data if d['details']['task'] == 'molecule_structure_prediction'],
            'reaction_prediction': [d for d in data if d['details']['task'] == 'reaction_prediction'],
            'retrosynthesis': [d for d in data if d['details']['task'] == 'retrosynthesis'],
            'balancing_chemical_equation': [d for d in data if d['details']['task'] == 'balancing_chemical_equation'],
            #'chemical_calculation': [d for d in data if d['details']['task'] == 'sci_calculate' and d['domain'] == 'Chemistry'],
            # L4
            'chemical_harmful_QA': [d for d in data if d['details']['task'] == 'harmful_QA' and d['domain'] == 'Chemistry'],
            'mol_toxicity_prediction': [d for d in data if d['details']['task'] == 'mol_toxicity_prediction'],
            'chemical_laboratory_safety_test': [d for d in data if d['details']['task'] == 'laboratory_safety_test' and d['domain'] == 'Chemistry'],
            # L5
            #'molecule_generation': [d for d in data if d['details']['task'] == 'molecule_generation'],
            'chemical_procedure_generation': [d for d in data if d['details']['task'] == 'procedure_generation' and d['domain'] == 'Chemistry'],
            'chemical_reagent_generation': [d for d in data if d['details']['task'] == 'reagent_generation' and d['domain'] == 'Chemistry'],
            ### Biology
            # L1
            #'protein_property_identification': [d for d in data if d['details']['task'] == 'protein_property_identification'],
            'biology_literature_QA': [d for d in data if d['details']['task'] == 'literature_multi_choice_question' and d['domain'] == 'Biology'],
            'protein_description_generation': [d for d in data if d['details']['task'] == 'protein_description_generation'],
            # L2
            'drug_drug_relation_extraction': [d for d in data if d['details']['subtask'] == 'drug_drug_relation_extraction'],
            'biomedical_judgment_and_interpretation': [d for d in data if d['details']['subtask'] == 'biomedical_judgment_and_interpretation'],
            'compound_disease_relation_extraction': [d for d in data if d['details']['subtask'] == 'compound_disease_relation_extraction'],
            #'gene_disease_relation_extraction': [d for d in data if d['details']['subtask'] == 'gene_disease_relation_extraction'],
            'biological_detailed_understanding': [d for d in data if d['details']['subtask'] == 'detailed_understanding' and d['domain'] == 'Biology'],
            'biological_text_summary': [d for d in data if d['details']['subtask'] == 'text_summary' and d['domain'] == 'Biology'],
            'biological_hypothesis_verification': [d for d in data if d['details']['subtask'] == 'hypothesis_verification' and d['domain'] == 'Biology'],
            #'biological_reasoning_and_interpretation': [d for d in data if d['details']['subtask'] == 'reasoning_and_interpretation' and d['domain'] == 'Biology'],
            # L3
            'solubility_prediction': [d for d in data if d['details']['subtask'] == 'solubility_prediction'],
            'beta_lactamase_activity_prediction': [d for d in data if d['details']['subtask'] == 'beta_lactamase_activity_prediction'],
            'fluorescence_prediction': [d for d in data if d['details']['subtask'] == 'fluorescence_prediction'],
            'GB1_ftness_prediction': [d for d in data if d['details']['subtask'] == 'GB1_ftness_prediction'],
            'stability_prediction': [d for d in data if d['details']['subtask'] == 'stability_prediction'],
            'Protein_Protein_Interaction': [d for d in data if d['details']['subtask'] == 'Protein_Protein_Interaction'],
            #'biological_calculation': [d for d in data if d['details']['task'] == 'sci_calculate' and d['domain'] == 'Biology'],
            # L4
            'biological_harmful_QA': [d for d in data if d['details']['task'] == 'harmful_QA' and d['domain'] == 'Biology'],
            'proteotoxicity_prediction': [d for d in data if d['details']['task'] == 'proteotoxicity_prediction'],
            'biological_laboratory_safety_test': [d for d in data if d['details']['task'] == 'laboratory_safety_test' and d['domain'] == 'Biology'],
            # L5
            'biological_procedure_generation': [d for d in data if d['details']['task'] == 'procedure_generation' and d['domain'] == 'Biology'],
            'biological_reagent_generation': [d for d in data if d['details']['task'] == 'reagent_generation' and d['domain'] == 'Biology'],
            #'protein_design': [d for d in data if d['details']['task'] == 'protein_design'],
            #'single_cell_analysis': [d for d in data if d['details']['task'] == 'single_cell_analysis'],
            ### Material
            # L1
            'material_literature_QA': [d for d in data if d['details']['task'] == 'material_literature_QA'],
            # L2
            'material_hypothesis_verification': [d for d in data if d['details']['subtask'] == 'material_hypothesis_verification'],
            'material_component_extraction': [d for d in data if d['details']['subtask'] == 'material_component_extraction'],
            'material_data_extraction': [d for d in data if d['details']['subtask'] == 'material_data_extraction'],
            'material_detailed_understanding': [d for d in data if d['details']['subtask'] == 'material_detailed_understanding'],
            #'material_reasoning_and_interpretation': [d for d in data if d['details']['subtask'] == 'material_reasoning_and_interpretation'],
            'material_text_summary': [d for d in data if d['details']['subtask'] == 'material_text_summary'],
            # L3
            'valence_electron_difference_calculation': [d for d in data if d['details']['task'] == 'valence_electron_difference_calculation'],
            #'material_calculation': [d for d in data if d['details']['task'] == 'material_calculation'],
            'lattice_volume_calculation': [d for d in data if d['details']['task'] == 'lattice_volume_calculation'],
            'perovskite_stability_prediction': [d for d in data if d['details']['task'] == 'perovskite_stability_prediction'],
            'diffusion_rate_analysis': [d for d in data if d['details']['task'] == 'diffusion_rate_analysis'],
            # L4
            'material_safety_QA': [d for d in data if d['details']['task'] == 'material_safety_QA'],
            'material_toxicity_prediction': [d for d in data if d['details']['task'] == 'material_toxicity_prediction'],
            # L5
            'crystal_structure_and_composition_analysis': [d for d in data if d['details']['task'] == 'crystal_structure_and_composition_analysis'],
            'specified_band_gap_material_generation': [d for d in data if d['details']['task'] == 'specified_band_gap_material_generation'],
            #'property_and_usage_analysis': [d for d in data if d['details']['task'] in ['property_and_usage_analysis', 'L5_material']],
            ### Physics
            # L1
            'physics_literature_QA': [d for d in data if d['details']['task'] == 'physics_literature_QA'],
            #'fundamental_physics_exam': [d for d in data if d['details']['task'] == 'fundamental_physics_exam'],
            # L2
            'physics_hypothesis_verification': [d for d in data if d['details']['subtask'] == 'physics_hypothesis_verification'],
            'physics_detailed_understanding': [d for d in data if d['details']['subtask'] == 'physics_detailed_understanding'],
            #'physics_reasoning_and_interpretation': [d for d in data if d['details']['subtask'] == 'physics_reasoning_and_interpretation'],
            'physics_text_summary': [d for d in data if d['details']['subtask'] == 'physics_text_summary'],
            # L3
            #'high_school_physics_calculation': [d for d in data if d['details']['task'] == 'high_school_physics_calculation'],
            'general_physics_calculation': [d for d in data if d['details']['task'] == 'general_physics_calculation'],
            'physics_formula_derivation': [d for d in data if d['details']['task'] == 'physics_formula_derivation'],
            # L4
            'physics_safety_QA': [d for d in data if d['details']['task'] == 'physics_safety_QA'],
            'physics_laboratory_safety_test': [d for d in data if d['details']['task'] == 'physics_laboratory_safety_test'],
            # L5
            'physics_problem_solving': [d for d in data if d['details']['task'] == 'physics_problem_solving'],
        }
        assert sum([len(d) for d in task_data.values()]) == len(data), f'length not equal, 0 length task: {[k for k, v in task_data.items() if len(v) == 0]}'
        print(">>>>>> Total data length:", len(data))
        return task_data
    except Exception as e:
        raise NotImplementedError(f"data error: {e}. please check your task name.")
