from modeling import LdaModeling


def main():

    # "https://raw.githubusercontent.com/selva86/datasets/master/newsgroups.json"
    # "https://datapane.com/u/khuyentran1401/reports/tweets/"

    num_topics = 3
    documents = None
    lda_instance = LdaModeling(documents)
    gensim_corpus, gensim_dictionary = lda_instance.preprocessing()
    lda_model = lda_instance.modeling(gensim_corpus, gensim_dictionary, num_topics)

    lda_instance.plotting(lda_model, gensim_corpus, gensim_dictionary)
    lda_instance.performance(lda_model, documents, gensim_corpus, gensim_dictionary)

    limit = 40
    start = 2
    step = 4
    model_list, coherence_values, optimal_model = lda_instance.plot_coherence_scores_versus_topics(
        documents, gensim_corpus, gensim_dictionary, limit, start, step
    )

    idx = 3
    optimal_model = lda_instance.pick_best_model(model_list, idx)

    dominant_topic_df = lda_instance.get_dominant_topic_of_sentence(optimal_model, gensim_corpus, documents)
    print(dominant_topic_df.head(10))

    topic_documents_df = lda_instance.get_most_representative_documents_for_topics(dominant_topic_df)
    print(topic_documents_df.head(10))

    df_dominant_topics = lda_instance.get_count_of_documents_for_topic(dominant_topic_df)
    print(df_dominant_topics.head(10))

    inp_df = None
    clusters_df = inp_df.groupby(dominant_topic_df['Dominant_Topic'].tolist())
    cluster_dict = {}
    for cluster, gdf in clusters_df:
        cluster_dict[cluster] = gdf['JobProfile'].tolist()
        
    for k, v in cluster_dict.items():
    #     if len(v)<=1:
    #         continue
        print(k)
        display(v[:10])
        print()

    print(lda_model.print_topic(23, topn=20))
if __name__ == "__main__":
    main()
