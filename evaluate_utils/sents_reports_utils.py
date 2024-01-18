import numpy as np
from collections import defaultdict
from dataset.constants import ANATOMICAL_REGIONS, REPORT_KEYS
import evaluate
def get_ref_sentences_for_selected_regions(reference_sentences, selected_regions):
    """
    Args:
        reference_sentences (List[List[str]]): outer list has len batch_size, inner list has len 29 (the inner list holds all reference phrases of a single image)
        selected_regions ([batch_size x 29]): boolean array that has exactly "num_regions_selected_in_batch" True values
    """
    # array of shape [batch_size x 29]
    reference_sentences = np.asarray(reference_sentences)

    ref_sentences_for_selected_regions = reference_sentences[selected_regions]

    return ref_sentences_for_selected_regions.tolist()


def get_sents_for_normal_abnormal_selected_regions(region_is_abnormal, selected_regions, generated_sentences_for_selected_regions, reference_sentences_for_selected_regions):
    selected_region_is_abnormal = region_is_abnormal[selected_regions]
    # selected_region_is_abnormal is a bool array of shape [num_regions_selected_in_batch] that specifies if a selected region is abnormal (True) or normal (False)

    gen_sents_for_selected_regions = np.asarray(generated_sentences_for_selected_regions)
    ref_sents_for_selected_regions = np.asarray(reference_sentences_for_selected_regions)

    gen_sents_for_normal_selected_regions = gen_sents_for_selected_regions[~selected_region_is_abnormal].tolist()
    gen_sents_for_abnormal_selected_regions = gen_sents_for_selected_regions[selected_region_is_abnormal].tolist()

    ref_sents_for_normal_selected_regions = ref_sents_for_selected_regions[~selected_region_is_abnormal].tolist()
    ref_sents_for_abnormal_selected_regions = ref_sents_for_selected_regions[selected_region_is_abnormal].tolist()

    return (
        gen_sents_for_normal_selected_regions,
        gen_sents_for_abnormal_selected_regions,
        ref_sents_for_normal_selected_regions,
        ref_sents_for_abnormal_selected_regions,
    )


bert_score = evaluate.load("bertscore")
def get_generated_reports(generated_sentences_for_selected_regions, selected_regions, sentence_tokenizer,
                          bertscore_threshold):
    """
    Args:
        generated_sentences_for_selected_regions (List[str]): of length "num_regions_selected_in_batch"
        selected_regions ([batch_size x 29]): boolean array that has exactly "num_regions_selected_in_batch" True values
        sentence_tokenizer: used in remove_duplicate_generated_sentences to separate the generated sentences

    Return:
        generated_reports (List[str]): list of length batch_size containing generated reports for every image in batch
        removed_similar_generated_sentences (List[Dict[str, List]): list of length batch_size containing dicts that map from one generated sentence to a list
        of other generated sentences that were removed because they were too similar. Useful for manually verifying if removing similar generated sentences was successful
    """

    def remove_duplicate_generated_sentences(gen_report_single_image, bert_score):
        def check_gen_sent_in_sents_to_be_removed(gen_sent, similar_generated_sents_to_be_removed):
            for lists_of_gen_sents_to_be_removed in similar_generated_sents_to_be_removed.values():
                if gen_sent in lists_of_gen_sents_to_be_removed:
                    return True

            return False

        # since different (closely related) regions can have the same generated sentence, we first remove exact duplicates

        # use sentence tokenizer to separate the generated sentences
        gen_sents_single_image = sentence_tokenizer(gen_report_single_image).sents

        # convert spacy.tokens.span.Span object into str by using .text attribute
        gen_sents_single_image = [sent.text for sent in gen_sents_single_image]

        # remove exact duplicates using a dict as an ordered set
        # note that dicts are insertion ordered as of Python 3.7
        gen_sents_single_image = list(dict.fromkeys(gen_sents_single_image))

        # there can still be generated sentences that are not exact duplicates, but nonetheless very similar
        # e.g. "The cardiomediastinal silhouette is normal." and "The cardiomediastinal silhouette is unremarkable."
        # to remove these "soft" duplicates, we use bertscore

        # similar_generated_sents_to_be_removed maps from one sentence to a list of similar sentences that are to be removed
        similar_generated_sents_to_be_removed = defaultdict(list)

        # TODO:
        # the nested for loops below check each generated sentence with every other generated sentence
        # this is not particularly efficient, since e.g. generated sentences for the region "right lung" most likely
        # will never be similar to generated sentences for the region "abdomen"
        # thus, one could speed up these checks by only checking anatomical regions that are similar to each other

        for i in range(len(gen_sents_single_image)):
            gen_sent_1 = gen_sents_single_image[i]

            for j in range(i + 1, len(gen_sents_single_image)):
                if check_gen_sent_in_sents_to_be_removed(gen_sent_1, similar_generated_sents_to_be_removed):
                    break

                gen_sent_2 = gen_sents_single_image[j]
                if check_gen_sent_in_sents_to_be_removed(gen_sent_2, similar_generated_sents_to_be_removed):
                    continue

                bert_score_result = bert_score.compute(
                    lang="en", predictions=[gen_sent_1], references=[gen_sent_2], model_type="distilbert-base-uncased"
                )

                if bert_score_result["f1"][0] > bertscore_threshold:
                    # remove the generated similar sentence that is shorter
                    if len(gen_sent_1) > len(gen_sent_2):
                        similar_generated_sents_to_be_removed[gen_sent_1].append(gen_sent_2)
                    else:
                        similar_generated_sents_to_be_removed[gen_sent_2].append(gen_sent_1)

        gen_report_single_image = " ".join(
            sent for sent in gen_sents_single_image if
            not check_gen_sent_in_sents_to_be_removed(sent, similar_generated_sents_to_be_removed)
        )

        return gen_report_single_image, similar_generated_sents_to_be_removed


    generated_reports = []
    removed_similar_generated_sentences = []
    curr_index = 0
    sentence_number = len(REPORT_KEYS)
    batch_size = int(len(generated_sentences_for_selected_regions) / len(REPORT_KEYS))

    for i_batch in range(batch_size):
        # sum up all True values for a single row in the array (corresponing to a single image)
        num_selected_regions_single_image = sentence_number

        # use curr_index and num_selected_regions_single_image to index all generated sentences corresponding to a single image
        gen_sents_single_image = generated_sentences_for_selected_regions[
                                 curr_index: curr_index + num_selected_regions_single_image
                                 ]

        # update curr_index for next image
        curr_index += num_selected_regions_single_image

        # concatenate generated sentences of a single image to a continuous string gen_report_single_image
        gen_report_single_image = " ".join(sent for sent in gen_sents_single_image)

        gen_report_single_image, similar_generated_sents_to_be_removed = remove_duplicate_generated_sentences(
            gen_report_single_image, bert_score
        )

        generated_reports.append(gen_report_single_image)
        removed_similar_generated_sentences.append(similar_generated_sents_to_be_removed)

    return generated_reports, removed_similar_generated_sentences


def update_gen_sentences_with_corresponding_regions(
    gen_sentences_with_corresponding_regions,
    generated_sents_for_selected_regions,
    selected_regions
):
    """
    Args:
        gen_sentences_with_corresponding_regions (list[list[tuple[str, str]]]):
            len(outer_list)= (NUM_BATCHES_OF_GENERATED_REPORTS_TO_SAVE_TO_FILE * BATCH_SIZE),
            and inner list has len of how many regions were selected for a given image.
            Inner list hold tuples of (region_name, gen_sent), i.e. region name and its corresponding generated sentence
        generated_sentences_for_selected_regions (List[str]): of length "num_regions_selected_in_batch"
        selected_regions ([batch_size x 29]): boolean array that has exactly "num_regions_selected_in_batch" True values
    """
    # def get_region_name(region_index: int):
    #     for i, region_name in enumerate(ANATOMICAL_REGIONS):
    #         if i == region_index:
    #             return region_name
    index_gen_sentence = 0
    batch_size = int(len(generated_sents_for_selected_regions) / len(REPORT_KEYS))
    # selected_regions_single_image is a row with 29 bool values corresponding to a single image
    for i_batch in range(batch_size):
        gen_sents_with_regions_single_image = []
        for i_report, report_key in enumerate(REPORT_KEYS):
            region_name = report_key
            gen_sent = generated_sents_for_selected_regions[index_gen_sentence]
            gen_sents_with_regions_single_image.append((region_name, gen_sent))
            index_gen_sentence += 1

        gen_sentences_with_corresponding_regions.append(gen_sents_with_regions_single_image)
        

def update_num_generated_sentences_per_image(
    gen_and_ref_sentences: dict,
    selected_regions: np.array
):
    """
    selected_regions is a boolean array of shape (batch_size x 29) that will have a True value for all regions that were selected and hence for which sentences were generated.
    Thus to get the number of generated sentences per image, we just have to add up the True value along axis 1 (i.e. along the region dimension)
    """
    num_gen_sents_per_image = selected_regions.sum(axis=1).tolist()  # indices is a list[int] of len(batch_size)
    gen_and_ref_sentences["num_generated_sentences_per_image"].extend([len(REPORT_KEYS)] * len(num_gen_sents_per_image))


def update_gen_and_ref_sentences_for_regions(
    gen_and_ref_sentences: dict,
    generated_sents_for_selected_regions: list[str],
    reference_sents_for_selected_regions: list[str],
    selected_regions: np.array
):

    for curr_index, [gen_sent, ref_sent] in enumerate(zip(generated_sents_for_selected_regions, reference_sents_for_selected_regions)):
        report_index = curr_index % len(REPORT_KEYS)
        curr_key = REPORT_KEYS[report_index]
        gen_and_ref_sentences[curr_key]["generated_sentences"].append(gen_sent)
        gen_and_ref_sentences[curr_key]["reference_sentences"].append(ref_sent)
