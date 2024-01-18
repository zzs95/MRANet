import os
# from configs.feat_text_generator_config import (
#     BATCH_SIZE,
#     NUM_BEAMS,
#     MAX_NUM_TOKENS_GENERATE,
#     NUM_BATCHES_OF_GENERATED_SENTENCES_TO_SAVE_TO_FILE,
#     NUM_BATCHES_OF_GENERATED_REPORTS_TO_SAVE_TO_FILE)
    
def write_sentences_and_reports_to_file(
    gen_and_ref_sentences,
    gen_and_ref_reports,
    gen_sentences_with_corresponding_regions,
    generated_sentences_and_reports_folder_path,
    overall_steps_taken,
):
    def write_sentences():
        txt_file_name = os.path.join(generated_sentences_and_reports_folder_path, "generated_sentences", f"generated_sentences_step_{overall_steps_taken}.txt")

        with open(txt_file_name, "w") as f:
            for gen_sent, ref_sent in zip(generated_sentences, reference_sentences):
                f.write(f"Generated sentence: {gen_sent}\n")
                f.write(f"Reference sentence: {ref_sent}\n\n")


    def write_reports():
        txt_file_name = os.path.join(
            generated_sentences_and_reports_folder_path,
            "generated_reports",
            f"generated_reports_step_{overall_steps_taken}.txt",
        )

        with open(txt_file_name, "a") as f:
            for gen_report, ref_report, removed_similar_gen_sents, gen_sents_with_regions_single_report in zip(
                generated_reports,
                reference_reports,
                removed_similar_generated_sentences,
                gen_sentences_with_corresponding_regions
            ):
                f.write(f"Generated report: {gen_report}\n\n")
                f.write(f"Reference report: {ref_report}\n\n")

                f.write("Generated sentences with their regions:\n")
                for region_name, gen_sent in gen_sents_with_regions_single_report:
                    f.write(f"\t{region_name}: {gen_sent}\n")
                f.write("\n")

                f.write("Generated sentences that were removed:\n")
                for gen_sent, list_similar_gen_sents in removed_similar_gen_sents.items():
                    f.write(f"\t{gen_sent} == {list_similar_gen_sents}\n")
                f.write("\n")

                f.write("=" * 30)
                f.write("\n\n")

    num_generated_sentences_to_save = NUM_BATCHES_OF_GENERATED_SENTENCES_TO_SAVE_TO_FILE * BATCH_SIZE
    num_generated_reports_to_save = NUM_BATCHES_OF_GENERATED_REPORTS_TO_SAVE_TO_FILE * BATCH_SIZE

    # all below are list of str
    generated_sentences = gen_and_ref_sentences["generated_sentences"][:num_generated_sentences_to_save]
    reference_sentences = gen_and_ref_sentences["reference_sentences"][:num_generated_sentences_to_save]

    write_sentences()

    # all below are list of str except removed_similar_generated_sentences which is a list of dict
    generated_reports = gen_and_ref_reports["generated_reports"][:num_generated_reports_to_save]
    reference_reports = gen_and_ref_reports["reference_reports"][:num_generated_reports_to_save]
    removed_similar_generated_sentences = gen_and_ref_reports["removed_similar_generated_sentences"][:num_generated_reports_to_save]

    write_reports()


def write_all_losses_and_scores_to_tensorboard(
    writer,
    overall_steps_taken,
    train_losses_dict,
    val_losses_dict,
    # obj_detector_scores,
    # region_selection_scores,
    # region_abnormal_scores,
    language_model_scores,
    current_lr
):
    def write_losses():
        for loss_type in train_losses_dict:
            writer.add_scalars(
                "_loss",
                {f"{loss_type}_train": train_losses_dict[loss_type], f"{loss_type}_val": val_losses_dict[loss_type]},
                overall_steps_taken,
            )

    def write_clinical_efficacy_scores(ce_score_dict):
        """
        ce_score_dict is of the structure:

        {
            precision_micro_5: ...,
            precision_micro_all: ...,
            precision_example_all: ...,
            recall_micro_5: ...,
            recall_micro_all: ...,
            recall_example_all: ...,
            f1_micro_5: ...,
            f1_micro_all: ...,
            f1_example_all: ...,
            acc_micro_5: ...,
            acc_micro_all: ...,
            acc_example_all: ...,
            condition_1 : {
                precision: ...,
                recall: ...,
                f1: ...,
                acc: ...,
            },
            condition_2 : {
                precision: ...,
                recall: ...,
                f1: ...,
                acc: ...,
            },
            ...,
            condition_14 : {
                precision: ...,
                recall: ...,
                f1: ...,
                acc: ...,
            }
        }

        where the "..." after the 4 metrics are the corresponding scores,
        and condition_* are from the 14 conditions in src/CheXbert/src/constants.py
        """
        for k, v in ce_score_dict.items():
            if k.startswith("precision") or k.startswith("recall") or k.startswith("f1") or k.startswith("acc"):
                writer.add_scalar(f"language_model/report/CE/{k}", v, overall_steps_taken)
            else:
                # k is a condition
                condition_name = "_".join(k.lower().split())
                for metric, score in ce_score_dict[k].items():
                    writer.add_scalar(f"language_model/report/CE/{condition_name}/{metric}", score, overall_steps_taken)

    def write_language_model_scores():
        """
        language_model_scores is a dict with keys:
            - all: for all generated sentences
            - normal: for all generated sentences corresponding to normal regions
            - abnormal: for all generated sentences corresponding to abnormal regions
            - report: for all generated reports
            - region: for generated sentences per region
        """
        for subset in language_model_scores:
            if subset == "region":
                for region_name in language_model_scores["region"]:
                    for metric, score in language_model_scores["region"][region_name].items():
                        # replace white space by underscore for region name (i.e. "right upper lung" -> "right_upper_lung")
                        region_name_underscored = "_".join(region_name.split())
                        writer.add_scalar(f"language_model/region/{region_name_underscored}/{metric}", score, overall_steps_taken)
            else:
                for metric, score in language_model_scores[subset].items():
                    if metric == "CE":
                        ce_score_dict = language_model_scores["report"]["CE"]
                        write_clinical_efficacy_scores(ce_score_dict)
                    else:
                        writer.add_scalar(f"language_model/{subset}/{metric}", score, overall_steps_taken)

    write_losses()
    if language_model_scores != None:
        write_language_model_scores()
    writer.add_scalar("lr", current_lr, overall_steps_taken)
    


def write_all_losses_and_scores_to_tensorboard_feat(
    writer,
    overall_steps_taken,
    train_losses_dict=None,
    val_losses_dict=None,
    current_lr=None
):
    def write_val_losses():
        for loss_type in val_losses_dict:
            writer.add_scalars(
                "_loss",
                {f"{loss_type}_val": val_losses_dict[loss_type]},
                overall_steps_taken,
            )
    def write_train_losses():
        for loss_type in train_losses_dict:
            writer.add_scalars(
                "_loss",
                {f"{loss_type}_train": train_losses_dict[loss_type]},
                overall_steps_taken,
            )
    if val_losses_dict != None:
        write_val_losses()
    if train_losses_dict != None:
        write_train_losses()
    if current_lr!=None:
        writer.add_scalar("lr", current_lr, overall_steps_taken)
        
        

def write_sentences_and_reports_to_file_for_test_set(
    gen_and_ref_sentences,
    gen_and_ref_reports,
    gen_sentences_with_corresponding_regions,
    path_test_set_evaluation_scores_txt_files,
    set_name=''
):
    def write_sentences():
        txt_file_name = os.path.join(path_test_set_evaluation_scores_txt_files, set_name+"_generated_sentences.txt")
        with open(txt_file_name, "a") as f:
            for gen_sent, ref_sent in zip(generated_sentences, reference_sentences):
                f.write(f"Generated sentence: {gen_sent}\n")
                f.write(f"Reference sentence: {ref_sent}\n\n")

    def write_reports():
        txt_file_name = os.path.join(path_test_set_evaluation_scores_txt_files, set_name+"_generated_reports.txt")
        with open(txt_file_name, "a") as f:
            for gen_report, ref_report, removed_similar_gen_sents, gen_sents_with_regions_single_report in zip(
                generated_reports,
                reference_reports,
                removed_similar_generated_sentences,
                gen_sentences_with_corresponding_regions
            ):
                f.write(f"Generated report: {gen_report}\n\n")
                f.write(f"Reference report: {ref_report}\n\n")

                f.write("Generated sentences with their regions:\n")
                for region_name, gen_sent in gen_sents_with_regions_single_report:
                    f.write(f"\t{region_name}: {gen_sent}\n")
                f.write("\n")

                f.write("Generated sentences that were removed:\n")
                for gen_sent, list_similar_gen_sents in removed_similar_gen_sents.items():
                    f.write(f"\t{gen_sent} == {list_similar_gen_sents}\n")
                f.write("\n")

                f.write("=" * 30)
                f.write("\n\n")

    # all below are list of str
    generated_sentences = gen_and_ref_sentences["generated_sentences"]
    reference_sentences = gen_and_ref_sentences["reference_sentences"]
    write_sentences()

    # all below are list of str except removed_similar_generated_sentences which is a list of dict
    generated_reports = gen_and_ref_reports["generated_reports"]
    reference_reports = gen_and_ref_reports["reference_reports"]
    removed_similar_generated_sentences = gen_and_ref_reports["removed_similar_generated_sentences"]

    write_reports()